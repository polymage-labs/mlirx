//===- AffineVectorize.cpp - Vectorize Pass Impl --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a new vectorizer for affine loop nests that is able to 
// perform inner or outer loop vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-vect"

using namespace mlir;

using llvm::SetVector;

static llvm::cl::OptionCategory clOptionsCategory("affine vectorize options");

static llvm::cl::opt<unsigned>
    clSimdWidth("affine-vectorize-simd-width",
                llvm::cl::desc("Hardware vector width for vectorization"),
                llvm::cl::init(256), llvm::cl::cat(clOptionsCategory));

namespace {

/// Base state for the vectorize pass.
/// Command line arguments are preempted by non-empty pass arguments.
struct Vectorize : public FunctionPass<Vectorize> {
  Vectorize();
  Vectorize(ArrayRef<int64_t> vectorSizes);
  void runOnFunction() override;

  // The vector widths.
  SmallVector<int64_t, 4> vectorSizes;

  static const unsigned kDefaultVectorWidth = 4;
};

} // end anonymous namespace

Vectorize::Vectorize() {}

/// Given an input type, provides a vector type for it of the provided width.
static VectorType getVectorizedType(Type inputType, unsigned width) {
  assert(width > 1 && "unexpected vector width");
  Type baseEltType = inputType;
  SmallVector<int64_t, 4> vecShape;
  if (auto vecEltType = inputType.dyn_cast<VectorType>()) {
    baseEltType = vecEltType.getElementType();
    vecShape.reserve(vecShape.size() + vecEltType.getRank());
    vecShape.assign(vecEltType.getShape().begin(), vecEltType.getShape().end());
  }
  vecShape.push_back(width);
  return VectorType::get(vecShape, baseEltType);
}

/// Casts a given input memref, uses memref_shape_cast op to cast it to a memref
/// with an elemental type that is `vector width` times (for eg., f32 becomes
/// vector<8xf32>, vector<8xf32> becomes vector<8x8xf32> if `vectorWidth` were
/// to be 8).
static Value *createVectorMemRef(Value *scalMemRef, unsigned vectorWidth) {
  auto scalMemRefType = scalMemRef->getType().cast<MemRefType>();
  auto shape = scalMemRefType.getShape();
  assert(shape.back() % vectorWidth == 0 && "unexpected memref shape");

  OpBuilder b(scalMemRef->getContext());
  if (auto *defOp = scalMemRef->getDefiningOp())
    b.setInsertionPointAfter(defOp);
  else
    b.setInsertionPointToStart(cast<BlockArgument>(scalMemRef)->getOwner());

  auto vecMemRefEltType =
      getVectorizedType(scalMemRefType.getElementType(), vectorWidth);

  SmallVector<int64_t, 4> vecMemRefShape(shape.begin(), shape.end());
  vecMemRefShape.back() /= vectorWidth;

  auto vecMemRefType = MemRefType::get(vecMemRefShape, vecMemRefEltType);
  return b.create<MemRefShapeCastOp>(b.getUnknownLoc(), vecMemRefType,
                                     scalMemRef);
}

static void getLiveInScalars(AffineForOp forOp,
                             SmallVectorImpl<Value *> &scalars) {
  forOp.walk([&](Operation *op) {
    for (auto *value : op->getOperands()) {
      auto *defOp = value->getDefiningOp();
      if (!defOp) {
        scalars.push_back(value);
        continue;
      }
      SmallVector<AffineForOp, 4> ivs;
      getLoopIVs(*defOp, &ivs);
      if (llvm::find(ivs, forOp) == ivs.end())
        scalars.push_back(value);
    }
  });
}

/// Vectorize any operation other than AffineLoadOp, AffineStoreOp,
/// and splat op. Operands of the op should have already been vectorized. The op
/// can't have any regions.
static Operation *vectorizeMiscLeafOp(Operation *op, unsigned width) {
  // Sanity checks.
  assert(!isa<AffineLoadOp>(op) &&
         "all loads should have already been fully vectorized");
  assert(!isa<AffineStoreOp>(op) &&
         "all stores should have already been fully vectorized");

  if (op->getNumRegions() != 0)
    return nullptr;

  SmallVector<Type, 8> vectorTypes;
  for (auto *v : op->getResults()) {
    vectorTypes.push_back(getVectorizedType(v->getType(), width));
  }
  SmallVector<Value *, 8> vectorOperands(op->getOperands());

  // Check whether a single operand is null. If so, vectorization failed.
  bool success = llvm::all_of(
      vectorOperands, [](Value *v) { return v->getType().isa<VectorType>(); });
  if (!success) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n[affine-vect]+++++ operands shoulds have been vectorized");
    return nullptr;
  }

  OpBuilder b(op);
  OperationState newOp(op->getLoc(), op->getName().getStringRef(),
                       vectorOperands, vectorTypes, op->getAttrs(),
                       /*successors=*/{},
                       /*regions=*/{}, op->hasResizableOperandsList());
  return b.createOperation(newOp);
}

/// Returns an affine map with the last result of `input' scaled down by
/// `factor'.
static AffineMap scaleDownLastResult(AffineMap input, int64_t factor) {
  SmallVector<AffineExpr, 4> results(input.getResults().begin(),
                                     input.getResults().end());
  results.back() = results.back().floorDiv(factor);
  return AffineMap::get(input.getNumDims(), input.getNumSymbols(), results);
}

/// Vectorizes a loop (either outer or inner, with a perfect or imperfectly
/// nested body).
LogicalResult mlir::loopVectorize(AffineForOp forOp,
                                  DenseMap<Value *, Value *> *vecMemRefMap) {
  // Walk and collect all memrefs that need to be turned into vector types (or
  // to higher dimensional vector types).
  //
  // For vector memrefs, loads are replaced; for stores, just operands is
  // replaced. For invariant load/stores, splat result of the loads; leave
  // stores alone if the store value is a scalar; otherwise, write the last
  // value.
  // Live-in scalars are splat. All other ops' operands are automatically
  // replaced as a result of the above. Replace such ops with new ones so that
  // their result types are vector types.
  //
  DenseSet<Operation *> toVecLoadOps, toVecStoreOps;
  SmallVector<Operation *, 4> toSplatLoadOps, writeLastEltStoreOps;
  // Mapping from a memref to its vector counterpart.
  DenseMap<Value *, Value *> toVecMemRefMap;
  SetVector<Value *> toVecMemRefs;

  // Analysis phase.
  bool error = false;

  forOp.walk([&](Operation *op) {
    auto loadOp = dyn_cast<AffineLoadOp>(op);
    auto storeOp = dyn_cast<AffineStoreOp>(op);
    if (!loadOp && !storeOp)
      return WalkResult::advance();

    bool isInvariant = loadOp ? isInvariantAccess(loadOp, forOp)
                              : isInvariantAccess(storeOp, forOp);
    if (isInvariant) {
      if (loadOp)
        toSplatLoadOps.push_back(loadOp);
      else
        writeLastEltStoreOps.push_back(storeOp);
      return WalkResult::advance();
    }

    Value *memref = loadOp ? loadOp.getMemRef() : storeOp.getMemRef();

    if (loadOp)
      toVecLoadOps.insert(loadOp);
    else
      toVecStoreOps.insert(storeOp);

    if (toVecMemRefs.count(memref) == 0) {
      toVecMemRefs.insert(memref);
    }
    return WalkResult::advance();
  });

  if (error)
    return failure();

  if (toVecMemRefs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No memrefs to vectorize\n");
    return failure();
  }

  // Compute the width for vectorization.
  int vectorWidth = -1;
  for (auto *memref : toVecMemRefs) {
    auto memrefType = memref->getType().cast<MemRefType>();
    auto eltType = memrefType.getElementType();
    if (eltType.isa<VectorType>()) {
      LLVM_DEBUG(llvm::dbgs() << "code already vectorized?\n");
      return failure();
    }

    if (clSimdWidth % eltType.getIntOrFloatBitWidth() != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "scalar width does not divide h/w vector width\n");
      return failure();
    }
    unsigned thisVectorWidth = clSimdWidth / eltType.getIntOrFloatBitWidth();
    if (vectorWidth == -1) {
      vectorWidth = thisVectorWidth;
    } else {
      if (std::max<unsigned>(vectorWidth, thisVectorWidth) %
              std::min<unsigned>(vectorWidth, thisVectorWidth) !=
          0) {
        LLVM_DEBUG(llvm::dbgs() << "Different memrefs require widths that "
                                   "aren't multiples of each other\n");
        return failure();
      }
      vectorWidth = std::min<unsigned>(vectorWidth, thisVectorWidth);
    }
  }

  assert(vectorWidth > 0 && "valid vector width should have been found\n");
  LLVM_DEBUG(llvm::dbgs() << "Using vector width: " << vectorWidth << "\n");

  // TODO: Handle cleanups with view ops.
  if (getLargestDivisorOfTripCount(forOp) % vectorWidth != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "Trip count not known to be a multiple of vector width\n");
    return failure();
  }

  // Check if memref dim size is a multiple of the width.
  for (auto *memref : toVecMemRefs) {
    auto memrefType = memref->getType().cast<MemRefType>();
    int64_t lastDimSize = memrefType.getDimSize(memrefType.getRank() - 1);
    if (lastDimSize == -1 || lastDimSize % vectorWidth != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "memref dimension not multiple of vector width\n");
      LLVM_DEBUG(memrefType.dump());
      return failure();
    }
  }

  // Create vector memrefs for the ones that will have their load/stores
  // vectorized.
  for (auto *vecMemRef : toVecMemRefs) {
    toVecMemRefMap.insert(
        {vecMemRef, createVectorMemRef(vecMemRef, vectorWidth)});
  }

  // End of analysis phase.

  // Vectorize load ops with the loop being vectorized indexing the fastest
  // varying dimension of the memref. Turn the load into a load on its vector
  // memref cast, and scale down the last access by vector width.
  for (auto op : toVecLoadOps) {
    auto loadOp = cast<AffineLoadOp>(op);
    OpBuilder rewriter(loadOp);
    SmallVector<Value *, 4> mapOperands(loadOp.getMapOperands());
    auto vecLoadOp = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), toVecMemRefMap[loadOp.getMemRef()],
        scaleDownLastResult(loadOp.getAffineMap(), vectorWidth), mapOperands);
    loadOp.getOperation()->replaceAllUsesWith(vecLoadOp);
    loadOp.erase();
  }

  // Splat invariant load ops.
  for (auto op : toSplatLoadOps) {
    auto loadOp = cast<AffineLoadOp>(op);
    OpBuilder rewriter(loadOp.getContext());
    rewriter.setInsertionPointAfter(loadOp);
    auto splat = rewriter.create<SplatOp>(
        loadOp.getLoc(),
        getVectorizedType(loadOp.getMemRefType().getElementType(), vectorWidth),
        loadOp.getResult());
    SmallPtrSet<Operation *, 1> exceptions = {splat};
    replaceAllUsesExcept(loadOp, splat, exceptions);
  }

  // Vectorize store ops with the loop being vectorized indexing the fastest
  // varying dimension of the memref. Turn the store into a store on its vector
  // memref cast, and scale down the last access by vector width.
  for (auto op : toVecStoreOps) {
    auto storeOp = cast<AffineStoreOp>(op);
    OpBuilder rewriter(storeOp);
    SmallVector<Value *, 4> mapOperands(storeOp.getMapOperands());
    rewriter.create<AffineStoreOp>(
        storeOp.getLoc(), storeOp.getValueToStore(),
        toVecMemRefMap[storeOp.getMemRef()],
        scaleDownLastResult(storeOp.getAffineMap(), vectorWidth), mapOperands);
    storeOp.erase();
  }

  // Vectorize remaining ops.
  forOp.walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op) ||
        isa<AffineApplyOp>(op) || isa<SplatOp>(op) ||
        isa<AffineTerminatorOp>(op))
      return;
    if (auto *vecOp = vectorizeMiscLeafOp(op, vectorWidth)) {
      op->replaceAllUsesWith(vecOp);
      if (op->use_empty())
        op->erase();
    }
  });

  SmallVector<Value *, 4> liveInScalars;
  getLiveInScalars(forOp, liveInScalars);

  // TODO: Splat live-in scalars.

  assert(writeLastEltStoreOps.empty() && "unimplemented last write store ops");

  // Set the step.
  forOp.setStep(forOp.getStep() * vectorWidth);

  // TODO: an initial check should provide a guarantee that if we complete this
  // method, everything would be vectorized.

  // Compose any affine apply ops, fold ops, drop dead ops, and normalize
  // strided loops.
  auto *context = forOp.getContext();
  OwningRewritePatternList patterns;
  AffineForOp::getCanonicalizationPatterns(patterns, context);
  AffineLoadOp::getCanonicalizationPatterns(patterns, context);
  AffineStoreOp::getCanonicalizationPatterns(patterns, context);
  applyPatternsGreedily(forOp.getParentOfType<FuncOp>(), std::move(patterns));

  if (vecMemRefMap)
    *vecMemRefMap = std::move(toVecMemRefMap);

  return success();
}

// This uses a sufficient condition for vectorization. Checks for parallelism
// and access contiguity.
static bool isVectorizable(AffineForOp forOp) {
  // TODO: make this powerful; the utilities do exist. ``No dependence cycle
  // returning within vector width iterations'' is a more powerful condition.
  if (!isLoopParallel(forOp))
    return false;

  bool isVectorizable = true;
  forOp.walk([&](Operation *op) {
    // Splat ops whose input is a vector type aren't support. So, we
    // can't vectorize such loops for now.
    if (isa<SplatOp>(op)) {
      isVectorizable = false;
      return WalkResult::interrupt();
    }

    auto loadOp = dyn_cast<AffineLoadOp>(op);
    auto storeOp = dyn_cast<AffineStoreOp>(op);
    if (!loadOp && !storeOp)
      // We can vectorize anything else.
      return WalkResult::advance();

    // mlir::isContiguousAccess checks for contiguity (stride 1) as well as
    // invariance, and returns true for either. Fix this.
    int memrefDim;
    bool ret =
        loadOp
            ? isContiguousAccess(forOp.getInductionVar(), loadOp, &memrefDim)
            : isContiguousAccess(forOp.getInductionVar(), storeOp, &memrefDim);

    if (ret && (memrefDim == 0 || memrefDim == -1))
      return WalkResult::advance();
    isVectorizable = false;
    return WalkResult::interrupt();
  });

  return isVectorizable;
}

void Vectorize::runOnFunction() {
  FuncOp f = getFunction();

  llvm::DenseSet<Operation *> vectorizableLoops;
  f.walk([&vectorizableLoops](AffineForOp loop) {
    if (isVectorizable(loop)) {
      vectorizableLoops.insert(loop);
    }
  });

  for (auto loop : vectorizableLoops) {
    LLVM_DEBUG(llvm::dbgs() << "\n******************************************");
    LLVM_DEBUG(llvm::dbgs() << "\n******************************************");
    LLVM_DEBUG(llvm::dbgs() << "\n[affine-vect] Vectorizing loop\n");
    LLVM_DEBUG(loop->dump());
    loopVectorize(cast<AffineForOp>(loop));
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAffineVectorizePass() {
  return std::make_unique<Vectorize>();
}

static PassRegistration<Vectorize> pass("affine-vectorize",
                                        "Vectorize affine for ops");
