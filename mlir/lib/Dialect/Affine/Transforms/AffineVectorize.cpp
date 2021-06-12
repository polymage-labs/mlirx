//===- AffineVectorize.cpp - AffineVectorize Pass -------------------------===//
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

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-vect"

using namespace mlir;

namespace {

/// Base state for the vectorize pass.
/// Command line arguments are preempted by non-empty pass arguments.
struct AffineVectorize : public AffineVectorizeBase<AffineVectorize> {

/// Include the generated pass utilities.
#define GEN_PASS_AffineVectorize
#include "mlir/Dialect/Affine/Passes.h.inc"

  AffineVectorize();
  AffineVectorize(ArrayRef<int64_t> vectorSizes);
  void runOnFunction() override;

  // The vector widths.
  SmallVector<int64_t, 4> vectorSizes;
};

} // end anonymous namespace

AffineVectorize::AffineVectorize() {}

// This uses a sufficient condition for vectorization. Checks for parallelism
// and access contiguity.
static bool isVectorizable(AffineForOp forOp) {
  // TODO: make this powerful; the utilities do exist. ``No dependence cycle
  // returning within vector width iterations'' is a more powerful condition.
  if (!isLoopParallel(forOp))
    return false;

  bool isVectorizable = true;
  forOp.walk([&](Operation *op) {
    // Splat ops whose input is a vector type aren't supportted. So, we
    // can't vectorize such loops for now.
    if (isa<SplatOp>(op)) {
      isVectorizable = false;
      return WalkResult::interrupt();
    }

    // Call ops and standard load/store (as opposed to affine load/stores) will
    // not be vectorized.
    if (isa<CallOp>(op) || isa<memref::LoadOp>(op) ||
        isa<memref::StoreOp>(op)) {
      isVectorizable = false;
      return WalkResult::interrupt();
    }

    if (isa<AffineApplyOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Uncomposed affine apply - run canonicalize");
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

void AffineVectorize::runOnFunction() {
  FuncOp f = getFunction();

  llvm::DenseSet<Operation *> vectorizableLoops;
  f.walk([&vectorizableLoops](AffineForOp loop) {
    if (isVectorizable(loop)) {
      vectorizableLoops.insert(loop);
    }
  });

  for (Operation *loop : vectorizableLoops) {
    LLVM_DEBUG(llvm::dbgs() << "\n******************************************");
    LLVM_DEBUG(llvm::dbgs() << "\n******************************************");
    LLVM_DEBUG(llvm::dbgs() << "\n[affine-vect] Vectorizing loop\n");
    LLVM_DEBUG(loop->dump());
    (void)loopVectorize(cast<AffineForOp>(loop), clSimdWidth);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineVectorizePass() {
  return std::make_unique<AffineVectorize>();
}
