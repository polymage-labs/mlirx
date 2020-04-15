//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous loop transformation routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopUtils.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "LoopUtils"

using namespace mlir;
using llvm::SetVector;
using llvm::SmallMapVector;

namespace {
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};
} // namespace

/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
static void getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
                                     AffineMap &map,
                                     SmallVectorImpl<Value> &operands) {
  auto lbMap = forOp.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap.getNumResults() != 1) {
    map = AffineMap();
    return;
  }

  AffineMap tripCountMap;
  SmallVector<Value, 4> tripCountOperands;
  buildTripCountMapAndOperands(forOp, &tripCountMap, &tripCountOperands);

  // Sometimes the trip count cannot be expressed as an affine expression.
  if (!tripCountMap) {
    map = AffineMap();
    return;
  }

  OpBuilder b(forOp);
  auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap,
                                    forOp.getLowerBoundOperands());

  // For each upper bound expr, get the range.
  // Eg: affine.for %i = lb to min (ub1, ub2),
  // where tripCountExprs yield (tr1, tr2), we create affine.apply's:
  // lb + tr1 - tr1 % ufactor, lb + tr2 - tr2 % ufactor; the results of all
  // these affine.apply's make up the cleanup loop lower bound.
  SmallVector<AffineExpr, 4> bumpExprs(tripCountMap.getNumResults());
  SmallVector<Value, 4> bumpValues(tripCountMap.getNumResults());
  int64_t step = forOp.getStep();
  for (unsigned i = 0, e = tripCountMap.getNumResults(); i < e; i++) {
    auto tripCountExpr = tripCountMap.getResult(i);
    bumpExprs[i] = (tripCountExpr - tripCountExpr % unrollFactor) * step;
    auto bumpMap = AffineMap::get(tripCountMap.getNumDims(),
                                  tripCountMap.getNumSymbols(), bumpExprs[i]);
    bumpValues[i] =
        b.create<AffineApplyOp>(forOp.getLoc(), bumpMap, tripCountOperands);
  }

  SmallVector<AffineExpr, 4> newUbExprs(tripCountMap.getNumResults());
  for (unsigned i = 0, e = bumpExprs.size(); i < e; i++)
    newUbExprs[i] = b.getAffineDimExpr(0) + b.getAffineDimExpr(i + 1);

  operands.clear();
  operands.push_back(lb);
  operands.append(bumpValues.begin(), bumpValues.end());
  map = AffineMap::get(1 + tripCountMap.getNumResults(), 0, newUbExprs);
  // Simplify the map + operands.
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  // Remove any affine.apply's that became dead from the simplification above.
  for (auto v : bumpValues)
    if (v.use_empty())
      v.getDefiningOp()->erase();

  if (lb.use_empty())
    lb.erase();
}

/// Promotes the loop body of a forOp to its containing block if the forOp
/// was known to have a single iteration.
// TODO(bondhugula): extend this for arbitrary affine bounds.
LogicalResult mlir::promoteIfSingleIteration(AffineForOp forOp) {
  Optional<uint64_t> tripCount = getConstantTripCount(forOp);
  if (!tripCount || tripCount.getValue() != 1)
    return failure();

  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  // Replaces all IV uses to its single iteration value.
  auto iv = forOp.getInductionVar();
  auto *parentBlock = forOp.getOperation()->getBlock();
  if (!iv.use_empty()) {
    if (forOp.hasConstantLowerBound()) {
      OpBuilder topBuilder(forOp.getParentOfType<FuncOp>().getBody());
      auto constOp = topBuilder.create<ConstantIndexOp>(
          forOp.getLoc(), forOp.getConstantLowerBound());
      iv.replaceAllUsesWith(constOp);
    } else {
      auto lbOperands = forOp.getLowerBoundOperands();
      auto lbMap = forOp.getLowerBoundMap();
      OpBuilder builder(parentBlock, Block::iterator(forOp));
      if (lbMap == builder.getDimIdentityMap()) {
        // No need of generating an affine.apply.
        iv.replaceAllUsesWith(lbOperands[0]);
      } else {
        auto affineApplyOp =
            builder.create<AffineApplyOp>(forOp.getLoc(), lbMap, lbOperands);
        iv.replaceAllUsesWith(affineApplyOp);
      }
    }
  }
  // Move the loop body operations, except for its terminator, to the loop's
  // containing block.
  forOp.getBody()->back().erase();
  parentBlock->getOperations().splice(Block::iterator(forOp),
                                      forOp.getBody()->getOperations());
  forOp.erase();
  return success();
}

/// Promotes all single iteration 'for' ops in `f`, i.e., moves
/// their body into the containing Block.
void mlir::promoteSingleIterationLoops(FuncOp f) {
  // Gathers all innermost loops through a post order pruned walk.
  f.walk([](AffineForOp forOp) { promoteIfSingleIteration(forOp); });
}

/// Generates an affine.for op with the specified lower and upper bounds
/// while generating the right IV remappings to realize shifts for operations in
/// its body. The operations that go into the loop body are specified in
/// opGroupQueue starting from the specified offset, and in that order. The
/// first element of the pair specifies the shift applied to that group of
/// operations; the shift is multiplied by the loop step before being applied.
/// Returns nullptr if the generated loop simplifies to a single iteration one.
static AffineForOp generateShiftedLoop(
    AffineMap lbMap, AffineMap ubMap,
    const std::vector<std::pair<uint64_t, ArrayRef<Operation *>>> &opGroupQueue,
    unsigned offset, AffineForOp srcForOp, OpBuilder b) {
  auto lbOperands = srcForOp.getLowerBoundOperands();
  auto ubOperands = srcForOp.getUpperBoundOperands();

  assert(lbMap.getNumInputs() == lbOperands.size());
  assert(ubMap.getNumInputs() == ubOperands.size());

  auto loopChunk = b.create<AffineForOp>(srcForOp.getLoc(), lbOperands, lbMap,
                                         ubOperands, ubMap, srcForOp.getStep());
  auto loopChunkIV = loopChunk.getInductionVar();
  auto srcIV = srcForOp.getInductionVar();

  BlockAndValueMapping operandMap;

  OpBuilder bodyBuilder = loopChunk.getBodyBuilder();
  for (auto it = opGroupQueue.begin() + offset, e = opGroupQueue.end(); it != e;
       ++it) {
    uint64_t shift = it->first;
    auto ops = it->second;
    // All 'same shift' operations get added with their operands being
    // remapped to results of cloned operations, and their IV used remapped.
    // Generate the remapping if the shift is not zero: remappedIV = newIV -
    // shift.
    if (!srcIV.use_empty() && shift != 0) {
      auto ivRemap = bodyBuilder.create<AffineApplyOp>(
          srcForOp.getLoc(),
          bodyBuilder.getSingleDimShiftAffineMap(
              -static_cast<int64_t>(srcForOp.getStep() * shift)),
          loopChunkIV);
      operandMap.map(srcIV, ivRemap);
    } else {
      operandMap.map(srcIV, loopChunkIV);
    }
    for (auto *op : ops)
      bodyBuilder.clone(*op, operandMap);
  };
  if (succeeded(promoteIfSingleIteration(loopChunk)))
    return AffineForOp();
  return loopChunk;
}

// The skewing of operations with respect to one another can be used for
// example to allow overlap of asynchronous operations (such as DMA
// communication) with computation, or just relative shifting of operations
// for better register reuse, locality or parallelism. As such, the shifts are
// typically expected to be at most of the order of the number of operations.
// This method should not be used as a substitute for loop distribution/fission.
// This method uses an algorithm// in time linear in the number of operations
// in the body of the for loop - (using the 'sweep line' paradigm). This method
// asserts preservation of SSA dominance. A check for that as well as that for
// memory-based dependence preservation check rests with the users of this
// method.
LogicalResult mlir::affineForOpBodySkew(AffineForOp forOp,
                                        ArrayRef<uint64_t> shifts,
                                        bool unrollPrologueEpilogue) {
  assert(forOp.getBody()->getOperations().size() == shifts.size() &&
         "too few/many shifts");
  if (forOp.getBody()->begin() == std::prev(forOp.getBody()->end()))
    return success();

  // If the trip counts aren't constant, we would need versioning and
  // conditional guards (or context information to prevent such versioning). The
  // better way to pipeline for such loops is to first tile them and extract
  // constant trip count "full tiles" before applying this.
  auto mayBeConstTripCount = getConstantTripCount(forOp);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(forOp.emitRemark("non-constant trip count loop not handled"));
    return success();
  }
  uint64_t tripCount = mayBeConstTripCount.getValue();

  assert(isOpwiseShiftValid(forOp, shifts) &&
         "shifts will lead to an invalid transformation\n");

  int64_t step = forOp.getStep();

  unsigned numChildOps = shifts.size();

  // Do a linear time (counting) sort for the shifts.
  uint64_t maxShift = *std::max_element(shifts.begin(), shifts.end());
  if (maxShift >= numChildOps) {
    // Large shifts are not the typical use case.
    forOp.emitWarning("not shifting because shifts are unrealistically large");
    return success();
  }

  // An array of operation groups sorted by shift amount; each group has all
  // operations with the same shift in the order in which they appear in the
  // body of the 'affine.for' op.
  std::vector<std::vector<Operation *>> sortedOpGroups(maxShift + 1);
  unsigned pos = 0;
  for (auto &op : forOp.getBody()->without_terminator()) {
    auto shift = shifts[pos++];
    sortedOpGroups[shift].push_back(&op);
  }

  // Unless the shifts have a specific pattern (which actually would be the
  // common use case), prologue and epilogue are not meaningfully defined.
  // Nevertheless, if 'unrollPrologueEpilogue' is set, we will treat the first
  // loop generated as the prologue and the last as epilogue and unroll these
  // fully.
  AffineForOp prologue, epilogue;

  // Do a sweep over the sorted shifts while storing open groups in a
  // vector, and generating loop portions as necessary during the sweep. A block
  // of operations is paired with its shift.
  std::vector<std::pair<uint64_t, ArrayRef<Operation *>>> opGroupQueue;

  auto origLbMap = forOp.getLowerBoundMap();
  uint64_t lbShift = 0;
  OpBuilder b(forOp);
  for (uint64_t d = 0, e = sortedOpGroups.size(); d < e; ++d) {
    // If nothing is shifted by d, continue.
    if (sortedOpGroups[d].empty())
      continue;
    if (!opGroupQueue.empty()) {
      assert(d > 0 &&
             "Queue expected to be empty when the first block is found");
      // The interval for which the loop needs to be generated here is:
      // [lbShift, min(lbShift + tripCount, d)) and the body of the
      // loop needs to have all operations in opQueue in that order.
      AffineForOp res;
      if (lbShift + tripCount * step < d * step) {
        res = generateShiftedLoop(
            b.getShiftedAffineMap(origLbMap, lbShift),
            b.getShiftedAffineMap(origLbMap, lbShift + tripCount * step),
            opGroupQueue, /*offset=*/0, forOp, b);
        // Entire loop for the queued op groups generated, empty it.
        opGroupQueue.clear();
        lbShift += tripCount * step;
      } else {
        res = generateShiftedLoop(b.getShiftedAffineMap(origLbMap, lbShift),
                                  b.getShiftedAffineMap(origLbMap, d),
                                  opGroupQueue, /*offset=*/0, forOp, b);
        lbShift = d * step;
      }

      if (res) {
        // Simplify/canonicalize the affine.for.
        OwningRewritePatternList patterns;
        AffineForOp::getCanonicalizationPatterns(patterns, res.getContext());
        bool erased;
        applyOpPatternsAndFold(res, std::move(patterns), &erased);

        if (!erased && !prologue)
          prologue = res;
        if (!erased)
          epilogue = res;
      }
    } else {
      // Start of first interval.
      lbShift = d * step;
    }
    // Augment the list of operations that get into the current open interval.
    opGroupQueue.push_back({d, sortedOpGroups[d]});
  }

  // Those operations groups left in the queue now need to be processed (FIFO)
  // and their loops completed.
  for (unsigned i = 0, e = opGroupQueue.size(); i < e; ++i) {
    uint64_t ubShift = (opGroupQueue[i].first + tripCount) * step;
    epilogue = generateShiftedLoop(b.getShiftedAffineMap(origLbMap, lbShift),
                                   b.getShiftedAffineMap(origLbMap, ubShift),
                                   opGroupQueue, /*offset=*/i, forOp, b);
    lbShift = ubShift;
    if (!prologue)
      prologue = epilogue;
  }

  // Erase the original for op.
  forOp.erase();

  if (unrollPrologueEpilogue && prologue)
    loopUnrollFull(prologue);
  if (unrollPrologueEpilogue && !epilogue && epilogue != prologue)
    loopUnrollFull(epilogue);

  return success();
}

// Collect perfectly nested loops starting from `rootForOps`.  Loops are
// perfectly nested if each loop is the first and only non-terminator operation
// in the parent loop.  Collect at most `maxLoops` loops and append them to
// `forOps`.
template <typename T>
static void getPerfectlyNestedLoopsImpl(
    SmallVectorImpl<T> &forOps, T rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    Block &body = rootForOp.region().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    rootForOp = dyn_cast<T>(&body.front());
    if (!rootForOp)
      return;
  }
}

/// Get perfectly nested sequence of loops starting at root of loop nest
/// (the first op being another AffineFor, and the second op - a terminator).
/// A loop is perfectly nested iff: the first op in the loop's body is another
/// AffineForOp, and the second op is a terminator).
void mlir::getPerfectlyNestedLoops(SmallVectorImpl<AffineForOp> &nestedLoops,
                                   AffineForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

void mlir::getPerfectlyNestedLoops(SmallVectorImpl<loop::ForOp> &nestedLoops,
                                   loop::ForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

/// Unrolls this loop completely.
LogicalResult mlir::loopUnrollFull(AffineForOp forOp) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (mayBeConstantTripCount.hasValue()) {
    uint64_t tripCount = mayBeConstantTripCount.getValue();
    if (tripCount == 1)
      return promoteIfSingleIteration(forOp);
    return loopUnrollByFactor(forOp, tripCount);
  }
  return failure();
}

/// Unrolls this loop by the specified factor or by the trip count (if constant)
/// whichever is lower.
LogicalResult mlir::loopUnrollUpToFactor(AffineForOp forOp,
                                         uint64_t unrollFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return loopUnrollByFactor(forOp, mayBeConstantTripCount.getValue());
  return loopUnrollByFactor(forOp, unrollFactor);
}

/// Unrolls this loop by the specified factor. Returns success if the loop
/// is successfully unrolled.
LogicalResult mlir::loopUnrollByFactor(AffineForOp forOp,
                                       uint64_t unrollFactor) {
  assert(unrollFactor > 0 && "unroll factor should be positive");

  if (unrollFactor == 1)
    return promoteIfSingleIteration(forOp);

  // Nothing in the loop body other than the terminator.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Loops where the lower bound is a max expression isn't supported for
  // unrolling since the trip count can be expressed as an affine function when
  // both the lower bound and the upper bound are multi-result maps. However,
  // one meaningful way to do such unrolling would be to specialize the loop for
  // the 'hotspot' case and unroll that hotspot.
  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  // If the trip count is lower than the unroll factor, no unrolled body.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return failure();

  // Generate the cleanup loop if trip count isn't a multiple of unrollFactor.
  if (getLargestDivisorOfTripCount(forOp) % unrollFactor != 0) {
    OpBuilder builder(forOp.getOperation()->getBlock(),
                      std::next(Block::iterator(forOp)));
    auto cleanupForOp = cast<AffineForOp>(builder.clone(*forOp));
    AffineMap cleanupMap;
    SmallVector<Value, 4> cleanupOperands;
    getCleanupLoopLowerBound(forOp, unrollFactor, cleanupMap, cleanupOperands);
    assert(cleanupMap &&
           "cleanup loop lower bound map for single result lower bound maps "
           "can always be determined");
    cleanupForOp.setLowerBound(cleanupOperands, cleanupMap);
    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupForOp);

    // Adjust upper bound of the original loop; this is the same as the lower
    // bound of the cleanup loop.
    forOp.setUpperBound(cleanupOperands, cleanupMap);
  }

  // Scale the step of loop being unrolled by unroll factor.
  int64_t step = forOp.getStep();
  forOp.setStep(step * unrollFactor);

  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  OpBuilder builder = forOp.getBodyBuilder();

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(forOp.getBody()->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  auto forOpIV = forOp.getInductionVar();
  for (unsigned i = 1; i < unrollFactor; i++) {
    BlockAndValueMapping operandMap;

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      // iv' = iv + 1/2/3...unrollFactor-1;
      auto d0 = builder.getAffineDimExpr(0);
      auto bumpMap = AffineMap::get(1, 0, {d0 + i * step});
      auto ivUnroll =
          builder.create<AffineApplyOp>(forOp.getLoc(), bumpMap, forOpIV);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = forOp.getBody()->begin(); it != std::next(srcBlockEnd);
         it++) {
      builder.clone(*it, operandMap);
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forOp);
  return success();
}

LogicalResult mlir::loopUnrollJamUpToFactor(AffineForOp forOp,
                                            uint64_t unrollJamFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return loopUnrollJamByFactor(forOp, mayBeConstantTripCount.getValue());
  return loopUnrollJamByFactor(forOp, unrollJamFactor);
}

/// Unrolls and jams this loop by the specified factor.
LogicalResult mlir::loopUnrollJamByFactor(AffineForOp forOp,
                                          uint64_t unrollJamFactor) {
  // Gathers all maximal sub-blocks of operations that do not themselves
  // include a for op (a operation could have a descendant for op though
  // in its tree).  Ignore the block terminators.
  struct JamBlockGatherer {
    // Store iterators to the first and last op of each sub-block found.
    std::vector<std::pair<Block::iterator, Block::iterator>> subBlocks;

    // This is a linear time walk.
    void walk(Operation *op) {
      for (auto &region : op->getRegions())
        for (auto &block : region)
          walk(block);
    }

    void walk(Block &block) {
      for (auto it = block.begin(), e = std::prev(block.end()); it != e;) {
        auto subBlockStart = it;
        while (it != e && !isa<AffineForOp>(&*it))
          ++it;
        if (it != subBlockStart)
          subBlocks.push_back({subBlockStart, std::prev(it)});
        // Process all for ops that appear next.
        while (it != e && isa<AffineForOp>(&*it))
          walk(&*it++);
      }
    }
  };

  assert(unrollJamFactor > 0 && "unroll jam factor should be positive");

  if (unrollJamFactor == 1)
    return promoteIfSingleIteration(forOp);

  // Nothing in the loop body other than the terminator.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Loops where both lower and upper bounds are multi-result maps won't be
  // unrolled (since the trip can't be expressed as an affine function in
  // general).
  // TODO(mlir-team): this may not be common, but we could support the case
  // where the lower bound is a multi-result map and the ub is a single result
  // one.
  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  // If the trip count is lower than the unroll jam factor, no unroll jam.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor) {
    LLVM_DEBUG(llvm::dbgs() << "[failed] trip count < unroll-jam factor\n");
    return failure();
  }

  // Gather all sub-blocks to jam upon the loop being unrolled.
  JamBlockGatherer jbg;
  jbg.walk(forOp);
  auto &subBlocks = jbg.subBlocks;

  // Generate the cleanup loop if trip count isn't a multiple of
  // unrollJamFactor.
  if (getLargestDivisorOfTripCount(forOp) % unrollJamFactor != 0) {
    // Insert the cleanup loop right after 'forOp'.
    OpBuilder builder(forOp.getOperation()->getBlock(),
                      std::next(Block::iterator(forOp)));
    auto cleanupAffineForOp = cast<AffineForOp>(builder.clone(*forOp));
    // Adjust the lower bound of the cleanup loop; its upper bound is the same
    // as the original loop's upper bound.
    AffineMap cleanupMap;
    SmallVector<Value, 4> cleanupOperands;
    getCleanupLoopLowerBound(forOp, unrollJamFactor, cleanupMap,
                             cleanupOperands);
    cleanupAffineForOp.setLowerBound(cleanupOperands, cleanupMap);

    // Promote the cleanup loop if it has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupAffineForOp);

    // Adjust the upper bound of the original loop - it will be the same as the
    // cleanup loop's lower bound. Its lower bound remains unchanged.
    forOp.setUpperBound(cleanupOperands, cleanupMap);
  }

  // Scale the step of loop being unroll-jammed by the unroll-jam factor.
  int64_t step = forOp.getStep();
  forOp.setStep(step * unrollJamFactor);

  auto forOpIV = forOp.getInductionVar();
  // Unroll and jam (appends unrollJamFactor - 1 additional copies).
  for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
    // Operand map persists across all sub-blocks.
    BlockAndValueMapping operandMap;
    for (auto &subBlock : subBlocks) {
      // Builder to insert unroll-jammed bodies. Insert right at the end of
      // sub-block.
      OpBuilder builder(subBlock.first->getBlock(), std::next(subBlock.second));

      // If the induction variable is used, create a remapping to the value for
      // this unrolled instance.
      if (!forOpIV.use_empty()) {
        // iv' = iv + i, i = 1 to unrollJamFactor-1.
        auto d0 = builder.getAffineDimExpr(0);
        auto bumpMap = AffineMap::get(1, 0, {d0 + i * step});
        auto ivUnroll =
            builder.create<AffineApplyOp>(forOp.getLoc(), bumpMap, forOpIV);
        operandMap.map(forOpIV, ivUnroll);
      }
      // Clone the sub-block being unroll-jammed.
      for (auto it = subBlock.first; it != std::next(subBlock.second); ++it)
        builder.clone(*it, operandMap);
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forOp);
  return success();
}

/// Performs loop interchange on 'forOpA' and 'forOpB', where 'forOpB' is
/// nested within 'forOpA' as the only non-terminator operation in its block.
void mlir::interchangeLoops(AffineForOp forOpA, AffineForOp forOpB) {
  assert(&*forOpA.getBody()->begin() == forOpB.getOperation());
  auto &forOpABody = forOpA.getBody()->getOperations();
  auto &forOpBBody = forOpB.getBody()->getOperations();

  // 1) Splice forOpA's non-terminator operations (which is just forOpB) just
  // before forOpA (in ForOpA's parent's block) this should leave 'forOpA's
  // body containing only the terminator.
  forOpA.getOperation()->getBlock()->getOperations().splice(
      Block::iterator(forOpA), forOpABody, forOpABody.begin(),
      std::prev(forOpABody.end()));
  // 2) Splice forOpB's non-terminator operations into the beginning of forOpA's
  // body (this leaves forOpB's body containing only the terminator).
  forOpABody.splice(forOpABody.begin(), forOpBBody, forOpBBody.begin(),
                    std::prev(forOpBBody.end()));
  // 3) Splice forOpA into the beginning of forOpB's body.
  forOpBBody.splice(forOpBBody.begin(),
                    forOpA.getOperation()->getBlock()->getOperations(),
                    Block::iterator(forOpA));
}

// Checks each dependence component against the permutation to see if the
// desired loop interchange would violate dependences by making the
// dependence component lexicographically negative.
static bool checkLoopInterchangeDependences(
    const std::vector<SmallVector<DependenceComponent, 2>> &depCompsVec,
    ArrayRef<AffineForOp> loops, ArrayRef<unsigned> loopPermMap) {
  // Invert permutation map.
  unsigned maxLoopDepth = loops.size();
  SmallVector<unsigned, 4> loopPermMapInv;
  loopPermMapInv.resize(maxLoopDepth);
  for (unsigned i = 0; i < maxLoopDepth; ++i)
    loopPermMapInv[loopPermMap[i]] = i;

  // Check each dependence component against the permutation to see if the
  // desired loop interchange permutation would make the dependence vectors
  // lexicographically negative.
  // Example 1: [-1, 1][0, 0]
  // Example 2: [0, 0][-1, 1]
  for (unsigned i = 0, e = depCompsVec.size(); i < e; ++i) {
    const SmallVector<DependenceComponent, 2> &depComps = depCompsVec[i];
    assert(depComps.size() >= maxLoopDepth);
    // Check if the first non-zero dependence component is positive.
    // This iterates through loops in the desired order.
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      unsigned permIndex = loopPermMapInv[j];
      assert(depComps[permIndex].lb.hasValue());
      int64_t depCompLb = depComps[permIndex].lb.getValue();
      if (depCompLb > 0)
        break;
      if (depCompLb < 0)
        return false;
    }
  }
  return true;
}

/// Checks if the loop interchange permutation 'loopPermMap' of the perfectly
/// nested sequence of loops in 'loops' would violate dependences.
bool mlir::isValidLoopInterchangePermutation(ArrayRef<AffineForOp> loops,
                                             ArrayRef<unsigned> loopPermMap) {
  // Gather dependence components for dependences between all ops in loop nest
  // rooted at 'loops[0]', at loop depths in range [1, maxLoopDepth].
  assert(loopPermMap.size() == loops.size());
  unsigned maxLoopDepth = loops.size();
  std::vector<SmallVector<DependenceComponent, 2>> depCompsVec;
  getDependenceComponents(loops[0], maxLoopDepth, &depCompsVec);
  return checkLoopInterchangeDependences(depCompsVec, loops, loopPermMap);
}

/// Return true if `loops` is a perfect nest.
static bool LLVM_ATTRIBUTE_UNUSED
isPerfectlyNested(ArrayRef<AffineForOp> loops) {
  auto outerLoop = loops.front();
  for (auto loop : loops.drop_front()) {
    auto parentForOp = dyn_cast<AffineForOp>(loop.getParentOp());
    // parentForOp's body should be just this loop and the terminator.
    if (parentForOp != outerLoop ||
        parentForOp.getBody()->getOperations().size() != 2)
      return false;
    outerLoop = loop;
  }
  return true;
}

// input[i] should move from position i -> permMap[i]. Returns the position in
// `input` that becomes the new outermost loop.
unsigned mlir::permuteLoops(MutableArrayRef<AffineForOp> input,
                            ArrayRef<unsigned> permMap) {
  assert(input.size() == permMap.size() && "invalid permutation map size");
  // Check whether the permutation spec is valid. This is a small vector - we'll
  // just sort and check if it's iota.
  SmallVector<unsigned, 4> checkPermMap(permMap.begin(), permMap.end());
  llvm::sort(checkPermMap);
  if (llvm::any_of(llvm::enumerate(checkPermMap),
                   [](const auto &en) { return en.value() != en.index(); }))
    assert(false && "invalid permutation map");

  // Nothing to do.
  if (input.size() < 2)
    return 0;

  assert(isPerfectlyNested(input) && "input not perfectly nested");

  // Compute the inverse mapping, invPermMap: since input[i] goes to position
  // permMap[i], position i of the permuted nest is at input[invPermMap[i]].
  SmallVector<std::pair<unsigned, unsigned>, 4> invPermMap;
  for (unsigned i = 0, e = input.size(); i < e; ++i)
    invPermMap.push_back({permMap[i], i});
  llvm::sort(invPermMap);

  // Move the innermost loop body to the loop that would be the innermost in the
  // permuted nest (only if the innermost loop is going to change).
  if (permMap.back() != input.size() - 1) {
    auto *destBody = input[invPermMap.back().second].getBody();
    auto *srcBody = input.back().getBody();
    destBody->getOperations().splice(destBody->begin(),
                                     srcBody->getOperations(), srcBody->begin(),
                                     std::prev(srcBody->end()));
  }

  // We'll move each loop in `input` in the reverse order so that its body is
  // empty when we are moving it; this incurs zero copies and no erasing.
  for (int i = input.size() - 1; i >= 0; --i) {
    // If this has to become the outermost loop after permutation, add it to the
    // parent block of the original root.
    if (permMap[i] == 0) {
      // If the root remains the same, nothing to do.
      if (i == 0)
        continue;
      // Make input[i] the new outermost loop moving it into parentBlock.
      auto *parentBlock = input[0].getOperation()->getBlock();
      parentBlock->getOperations().splice(
          Block::iterator(input[0]),
          input[i].getOperation()->getBlock()->getOperations(),
          Block::iterator(input[i]));
      continue;
    }

    // If the parent in the permuted order is the same as in the original,
    // nothing to do.
    unsigned parentPosInInput = invPermMap[permMap[i] - 1].second;
    if (i > 0 && static_cast<unsigned>(i - 1) == parentPosInInput)
      continue;

    // Move input[i] to its surrounding loop in the transformed nest.
    auto *destBody = input[parentPosInInput].getBody();
    destBody->getOperations().splice(
        destBody->begin(), input[i].getOperation()->getBlock()->getOperations(),
        Block::iterator(input[i]));
  }

  return invPermMap[0].second;
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
AffineForOp mlir::sinkSequentialLoops(AffineForOp forOp) {
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);
  if (loops.size() < 2)
    return forOp;

  // Gather dependence components for dependences between all ops in loop nest
  // rooted at 'loops[0]', at loop depths in range [1, maxLoopDepth].
  unsigned maxLoopDepth = loops.size();
  std::vector<SmallVector<DependenceComponent, 2>> depCompsVec;
  getDependenceComponents(loops[0], maxLoopDepth, &depCompsVec);

  // Mark loops as either parallel or sequential.
  SmallVector<bool, 8> isParallelLoop(maxLoopDepth, true);
  for (unsigned i = 0, e = depCompsVec.size(); i < e; ++i) {
    SmallVector<DependenceComponent, 2> &depComps = depCompsVec[i];
    assert(depComps.size() >= maxLoopDepth);
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      DependenceComponent &depComp = depComps[j];
      assert(depComp.lb.hasValue() && depComp.ub.hasValue());
      if (depComp.lb.getValue() != 0 || depComp.ub.getValue() != 0)
        isParallelLoop[j] = false;
    }
  }

  // Count the number of parallel loops.
  unsigned numParallelLoops = 0;
  for (unsigned i = 0, e = isParallelLoop.size(); i < e; ++i)
    if (isParallelLoop[i])
      ++numParallelLoops;

  // Compute permutation of loops that sinks sequential loops (and thus raises
  // parallel loops) while preserving relative order.
  SmallVector<unsigned, 4> loopPermMap(maxLoopDepth);
  unsigned nextSequentialLoop = numParallelLoops;
  unsigned nextParallelLoop = 0;
  for (unsigned i = 0; i < maxLoopDepth; ++i) {
    if (isParallelLoop[i]) {
      loopPermMap[i] = nextParallelLoop++;
    } else {
      loopPermMap[i] = nextSequentialLoop++;
    }
  }

  // Check if permutation 'loopPermMap' would violate dependences.
  if (!checkLoopInterchangeDependences(depCompsVec, loops, loopPermMap))
    return forOp;
  // Perform loop interchange according to permutation 'loopPermMap'.
  unsigned loopNestRootIndex = permuteLoops(loops, loopPermMap);
  return loops[loopNestRootIndex];
}

// Factors out common behavior to add a new `iv` (resp. `iv` + `offset`) to the
// lower (resp. upper) loop bound. When called for both the lower and upper
// bounds, the resulting IR resembles:
//
// ```mlir
//    affine.for %i = max (`iv, ...) to min (`iv` + `offset`) {
//      ...
//    }
// ```
static void augmentMapAndBounds(OpBuilder &b, Value iv, AffineMap *map,
                                SmallVector<Value, 4> *operands,
                                int64_t offset = 0) {
  auto bounds = llvm::to_vector<4>(map->getResults());
  bounds.push_back(b.getAffineDimExpr(map->getNumDims()) + offset);
  operands->insert(operands->begin() + map->getNumDims(), iv);
  *map = AffineMap::get(map->getNumDims() + 1, map->getNumSymbols(), bounds);
  canonicalizeMapAndOperands(map, operands);
}

// Stripmines `forOp` by `factor` and sinks it under each of the `targets`.
// Stripmine-sink is a primitive building block for generalized tiling of
// imperfectly nested loops.
// This transformation is purely mechanical and does not check legality,
// profitability or even structural correctness. It is the user's
// responsibility to specify `targets` that are dominated by `forOp`.
// Returns the new AffineForOps, one per `targets`, nested immediately under
// each of the `targets`.
static SmallVector<AffineForOp, 8>
stripmineSink(AffineForOp forOp, uint64_t factor,
              ArrayRef<AffineForOp> targets) {
  auto originalStep = forOp.getStep();
  auto scaledStep = originalStep * factor;
  forOp.setStep(scaledStep);

  OpBuilder b(forOp.getOperation()->getBlock(),
              std::next(Block::iterator(forOp)));

  // Lower-bound map creation.
  auto lbMap = forOp.getLowerBoundMap();
  SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
  augmentMapAndBounds(b, forOp.getInductionVar(), &lbMap, &lbOperands);

  // Upper-bound map creation.
  auto ubMap = forOp.getUpperBoundMap();
  SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());
  augmentMapAndBounds(b, forOp.getInductionVar(), &ubMap, &ubOperands,
                      /*offset=*/scaledStep);

  auto iv = forOp.getInductionVar();
  SmallVector<AffineForOp, 8> innerLoops;
  for (auto t : targets) {
    // Insert newForOp before the terminator of `t`.
    OpBuilder b = t.getBodyBuilder();
    auto newForOp = b.create<AffineForOp>(t.getLoc(), lbOperands, lbMap,
                                          ubOperands, ubMap, originalStep);
    auto begin = t.getBody()->begin();
    // Skip terminator and `newForOp` which is just before the terminator.
    auto nOps = t.getBody()->getOperations().size() - 2;
    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        t.getBody()->getOperations(), begin, std::next(begin, nOps));
    replaceAllUsesInRegionWith(iv, newForOp.getInductionVar(),
                               newForOp.region());
    innerLoops.push_back(newForOp);
  }

  return innerLoops;
}

static Loops stripmineSink(loop::ForOp forOp, Value factor,
                           ArrayRef<loop::ForOp> targets) {
  auto originalStep = forOp.step();
  auto iv = forOp.getInductionVar();

  OpBuilder b(forOp);
  forOp.setStep(b.create<MulIOp>(forOp.getLoc(), originalStep, factor));

  Loops innerLoops;
  for (auto t : targets) {
    // Save information for splicing ops out of t when done
    auto begin = t.getBody()->begin();
    auto nOps = t.getBody()->getOperations().size();

    // Insert newForOp before the terminator of `t`.
    OpBuilder b(t.getBodyBuilder());
    Value stepped = b.create<AddIOp>(t.getLoc(), iv, forOp.step());
    Value less = b.create<CmpIOp>(t.getLoc(), CmpIPredicate::slt,
                                  forOp.upperBound(), stepped);
    Value ub =
        b.create<SelectOp>(t.getLoc(), less, forOp.upperBound(), stepped);

    // Splice [begin, begin + nOps - 1) into `newForOp` and replace uses.
    auto newForOp = b.create<loop::ForOp>(t.getLoc(), iv, ub, originalStep);
    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        t.getBody()->getOperations(), begin, std::next(begin, nOps - 1));
    replaceAllUsesInRegionWith(iv, newForOp.getInductionVar(),
                               newForOp.region());

    innerLoops.push_back(newForOp);
  }

  return innerLoops;
}

// Stripmines a `forOp` by `factor` and sinks it under a single `target`.
// Returns the new AffineForOps, nested immediately under `target`.
template <typename ForType, typename SizeType>
static ForType stripmineSink(ForType forOp, SizeType factor, ForType target) {
  // TODO(ntv): Use cheap structural assertions that targets are nested under
  // forOp and that targets are not nested under each other when DominanceInfo
  // exposes the capability. It seems overkill to construct a whole function
  // dominance tree at this point.
  auto res = stripmineSink(forOp, factor, ArrayRef<ForType>{target});
  assert(res.size() == 1 && "Expected 1 inner forOp");
  return res[0];
}

template <typename ForType, typename SizeType>
static SmallVector<SmallVector<ForType, 8>, 8>
tileImpl(ArrayRef<ForType> forOps, ArrayRef<SizeType> sizes,
         ArrayRef<ForType> targets) {
  SmallVector<SmallVector<ForType, 8>, 8> res;
  SmallVector<ForType, 8> currentTargets(targets.begin(), targets.end());
  for (auto it : llvm::zip(forOps, sizes)) {
    auto step = stripmineSink(std::get<0>(it), std::get<1>(it), currentTargets);
    res.push_back(step);
    currentTargets = step;
  }
  return res;
}

SmallVector<SmallVector<AffineForOp, 8>, 8>
mlir::tile(ArrayRef<AffineForOp> forOps, ArrayRef<uint64_t> sizes,
           ArrayRef<AffineForOp> targets) {
  return tileImpl(forOps, sizes, targets);
}

SmallVector<Loops, 8> mlir::tile(ArrayRef<loop::ForOp> forOps,
                                 ArrayRef<Value> sizes,
                                 ArrayRef<loop::ForOp> targets) {
  return tileImpl(forOps, sizes, targets);
}

template <typename ForType, typename SizeType>
static SmallVector<ForType, 8>
tileImpl(ArrayRef<ForType> forOps, ArrayRef<SizeType> sizes, ForType target) {
  SmallVector<ForType, 8> res;
  for (auto loops : tile(forOps, sizes, ArrayRef<ForType>{target})) {
    assert(loops.size() == 1);
    res.push_back(loops[0]);
  }
  return res;
}

SmallVector<AffineForOp, 8> mlir::tile(ArrayRef<AffineForOp> forOps,
                                       ArrayRef<uint64_t> sizes,
                                       AffineForOp target) {
  return tileImpl(forOps, sizes, target);
}

Loops mlir::tile(ArrayRef<loop::ForOp> forOps, ArrayRef<Value> sizes,
                 loop::ForOp target) {
  return tileImpl(forOps, sizes, target);
}

Loops mlir::tilePerfectlyNested(loop::ForOp rootForOp, ArrayRef<Value> sizes) {
  // Collect perfectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<loop::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  return ::tile(forOps, sizes, forOps.back());
}

// Build the IR that performs ceil division of a positive value by a constant:
//    ceildiv(a, B) = divis(a + (B-1), B)
// where divis is rounding-to-zero division.
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             int64_t divisor) {
  assert(divisor > 0 && "expected positive divisor");
  assert(dividend.getType().isIndex() && "expected index-typed value");

  Value divisorMinusOneCst = builder.create<ConstantIndexOp>(loc, divisor - 1);
  Value divisorCst = builder.create<ConstantIndexOp>(loc, divisor);
  Value sum = builder.create<AddIOp>(loc, dividend, divisorMinusOneCst);
  return builder.create<SignedDivIOp>(loc, sum, divisorCst);
}

// Build the IR that performs ceil division of a positive value by another
// positive value:
//    ceildiv(a, b) = divis(a + (b - 1), b)
// where divis is rounding-to-zero division.
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             Value divisor) {
  assert(dividend.getType().isIndex() && "expected index-typed value");

  Value cstOne = builder.create<ConstantIndexOp>(loc, 1);
  Value divisorMinusOne = builder.create<SubIOp>(loc, divisor, cstOne);
  Value sum = builder.create<AddIOp>(loc, dividend, divisorMinusOne);
  return builder.create<SignedDivIOp>(loc, sum, divisor);
}

// Hoist the ops within `outer` that appear before `inner`.
// Such ops include the ops that have been introduced by parametric tiling.
// Ops that come from triangular loops (i.e. that belong to the program slice
// rooted at `outer`) and ops that have side effects cannot be hoisted.
// Return failure when any op fails to hoist.
static LogicalResult hoistOpsBetween(loop::ForOp outer, loop::ForOp inner) {
  SetVector<Operation *> forwardSlice;
  getForwardSlice(outer.getOperation(), &forwardSlice, [&inner](Operation *op) {
    return op != inner.getOperation();
  });
  LogicalResult status = success();
  SmallVector<Operation *, 8> toHoist;
  for (auto &op : outer.getBody()->without_terminator()) {
    // Stop when encountering the inner loop.
    if (&op == inner.getOperation())
      break;
    // Skip over non-hoistable ops.
    if (forwardSlice.count(&op) > 0) {
      status = failure();
      continue;
    }
    // Skip loop::ForOp, these are not considered a failure.
    if (op.getNumRegions() > 0)
      continue;
    // Skip other ops with regions.
    if (op.getNumRegions() > 0) {
      status = failure();
      continue;
    }
    // Skip if op has side effects.
    // TODO(ntv): loads to immutable memory regions are ok.
    if (!MemoryEffectOpInterface::hasNoEffect(&op)) {
      status = failure();
      continue;
    }
    toHoist.push_back(&op);
  }
  auto *outerForOp = outer.getOperation();
  for (auto *op : toHoist)
    op->moveBefore(outerForOp);
  return status;
}

// Traverse the interTile and intraTile loops and try to hoist ops such that
// bands of perfectly nested loops are isolated.
// Return failure if either perfect interTile or perfect intraTile bands cannot
// be formed.
static LogicalResult tryIsolateBands(const TileLoops &tileLoops) {
  LogicalResult status = success();
  auto &interTile = tileLoops.first;
  auto &intraTile = tileLoops.second;
  auto size = interTile.size();
  assert(size == intraTile.size());
  if (size <= 1)
    return success();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(intraTile[0], intraTile[s])
                               : failure();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(interTile[0], interTile[s])
                               : failure();
  return status;
}

TileLoops mlir::extractFixedOuterLoops(loop::ForOp rootForOp,
                                       ArrayRef<int64_t> sizes) {
  // Collect perfectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<loop::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  // Compute the tile sizes such that i-th outer loop executes size[i]
  // iterations.  Given that the loop current executes
  //   numIterations = ceildiv((upperBound - lowerBound), step)
  // iterations, we need to tile with size ceildiv(numIterations, size[i]).
  SmallVector<Value, 4> tileSizes;
  tileSizes.reserve(sizes.size());
  for (unsigned i = 0, e = sizes.size(); i < e; ++i) {
    assert(sizes[i] > 0 && "expected strictly positive size for strip-mining");

    auto forOp = forOps[i];
    OpBuilder builder(forOp);
    auto loc = forOp.getLoc();
    Value diff =
        builder.create<SubIOp>(loc, forOp.upperBound(), forOp.lowerBound());
    Value numIterations = ceilDivPositive(builder, loc, diff, forOp.step());
    Value iterationsPerBlock =
        ceilDivPositive(builder, loc, numIterations, sizes[i]);
    tileSizes.push_back(iterationsPerBlock);
  }

  // Call parametric tiling with the given sizes.
  auto intraTile = tile(forOps, tileSizes, forOps.back());
  TileLoops tileLoops = std::make_pair(forOps, intraTile);

  // TODO(ntv, zinenko) for now we just ignore the result of band isolation.
  // In the future, mapping decisions may be impacted by the ability to
  // isolate perfectly nested bands.
  tryIsolateBands(tileLoops);

  return tileLoops;
}

/// Return the new lower bound, upper bound, and step in that order. Insert any
/// additional bounds calculations before the given builder and any additional
/// conversion back to the original loop induction value inside the given Block.
static LoopParams normalizeLoop(OpBuilder &boundsBuilder,
                                OpBuilder &insideLoopBuilder, Location loc,
                                Value lowerBound, Value upperBound, Value step,
                                Value inductionVar) {
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto ubCst =
          dyn_cast_or_null<ConstantIndexOp>(lowerBound.getDefiningOp()))
    isZeroBased = ubCst.getValue() == 0;

  bool isStepOne = false;
  if (auto stepCst = dyn_cast_or_null<ConstantIndexOp>(step.getDefiningOp()))
    isStepOne = stepCst.getValue() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  // TODO(zinenko): introduce support for negative steps or emit dynamic asserts
  // on step positivity, whatever gets implemented first.
  if (isZeroBased && isStepOne)
    return {/*lowerBound=*/lowerBound, /*upperBound=*/upperBound,
            /*step=*/step};

  Value diff = boundsBuilder.create<SubIOp>(loc, upperBound, lowerBound);
  Value newUpperBound = ceilDivPositive(boundsBuilder, loc, diff, step);

  Value newLowerBound =
      isZeroBased ? lowerBound : boundsBuilder.create<ConstantIndexOp>(loc, 0);
  Value newStep =
      isStepOne ? step : boundsBuilder.create<ConstantIndexOp>(loc, 1);

  // Insert code computing the value of the original loop induction variable
  // from the "normalized" one.
  Value scaled =
      isStepOne ? inductionVar
                : insideLoopBuilder.create<MulIOp>(loc, inductionVar, step);
  Value shifted =
      isZeroBased ? scaled
                  : insideLoopBuilder.create<AddIOp>(loc, scaled, lowerBound);

  SmallPtrSet<Operation *, 2> preserve{scaled.getDefiningOp(),
                                       shifted.getDefiningOp()};
  replaceAllUsesExcept(inductionVar, shifted, preserve);
  return {/*lowerBound=*/newLowerBound, /*upperBound=*/newUpperBound,
          /*step=*/newStep};
}

/// Transform a loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1 {
///     %i = %ii * %s + %lb
/// Insert the induction variable remapping in the body of `inner`, which is
/// expected to be either `loop` or another loop perfectly nested under `loop`.
/// Insert the definition of new bounds immediate before `outer`, which is
/// expected to be either `loop` or its parent in the loop nest.
static void normalizeLoop(loop::ForOp loop, loop::ForOp outer,
                          loop::ForOp inner) {
  OpBuilder builder(outer);
  OpBuilder innerBuilder = OpBuilder::atBlockBegin(inner.getBody());
  auto loopPieces =
      normalizeLoop(builder, innerBuilder, loop.getLoc(), loop.lowerBound(),
                    loop.upperBound(), loop.step(), loop.getInductionVar());

  loop.setLowerBound(loopPieces.lowerBound);
  loop.setUpperBound(loopPieces.upperBound);
  loop.setStep(loopPieces.step);
}

void mlir::coalesceLoops(MutableArrayRef<loop::ForOp> loops) {
  if (loops.size() < 2)
    return;

  loop::ForOp innermost = loops.back();
  loop::ForOp outermost = loops.front();

  // 1. Make sure all loops iterate from 0 to upperBound with step 1.  This
  // allows the following code to assume upperBound is the number of iterations.
  for (auto loop : loops)
    normalizeLoop(loop, outermost, innermost);

  // 2. Emit code computing the upper bound of the coalesced loop as product
  // of the number of iterations of all loops.
  OpBuilder builder(outermost);
  Location loc = outermost.getLoc();
  Value upperBound = outermost.upperBound();
  for (auto loop : loops.drop_front())
    upperBound = builder.create<MulIOp>(loc, upperBound, loop.upperBound());
  outermost.setUpperBound(upperBound);

  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables.  For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  Value previous = outermost.getInductionVar();
  for (unsigned i = 0, e = loops.size(); i < e; ++i) {
    unsigned idx = loops.size() - i - 1;
    if (i != 0)
      previous = builder.create<SignedDivIOp>(loc, previous,
                                              loops[idx + 1].upperBound());

    Value iv = (i == e - 1) ? previous
                            : builder.create<SignedRemIOp>(
                                  loc, previous, loops[idx].upperBound());
    replaceAllUsesInRegionWith(loops[idx].getInductionVar(), iv,
                               loops.back().region());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  loop::ForOp second = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(second.getOperation()),
      innermost.getBody()->getOperations());
  second.erase();
}

void mlir::collapseParallelLoops(
    loop::ParallelOp loops,
    ArrayRef<std::vector<unsigned>> combinedDimensions) {
  OpBuilder outsideBuilder(loops);
  Location loc = loops.getLoc();

  // Normalize ParallelOp's iteration pattern.
  SmallVector<Value, 3> normalizedLowerBounds;
  SmallVector<Value, 3> normalizedSteps;
  SmallVector<Value, 3> normalizedUpperBounds;
  for (unsigned i = 0, e = loops.getNumLoops(); i < e; ++i) {
    OpBuilder insideLoopBuilder = OpBuilder::atBlockBegin(loops.getBody());
    auto resultBounds =
        normalizeLoop(outsideBuilder, insideLoopBuilder, loc,
                      loops.lowerBound()[i], loops.upperBound()[i],
                      loops.step()[i], loops.getBody()->getArgument(i));

    normalizedLowerBounds.push_back(resultBounds.lowerBound);
    normalizedUpperBounds.push_back(resultBounds.upperBound);
    normalizedSteps.push_back(resultBounds.step);
  }

  // Combine iteration spaces.
  SmallVector<Value, 3> lowerBounds;
  SmallVector<Value, 3> steps;
  SmallVector<Value, 3> upperBounds;
  auto cst0 = outsideBuilder.create<ConstantIndexOp>(loc, 0);
  auto cst1 = outsideBuilder.create<ConstantIndexOp>(loc, 1);
  for (unsigned i = 0, e = combinedDimensions.size(); i < e; ++i) {
    Value newUpperBound = outsideBuilder.create<ConstantIndexOp>(loc, 1);
    for (auto idx : combinedDimensions[i]) {
      newUpperBound = outsideBuilder.create<MulIOp>(loc, newUpperBound,
                                                    normalizedUpperBounds[idx]);
    }
    lowerBounds.push_back(cst0);
    steps.push_back(cst1);
    upperBounds.push_back(newUpperBound);
  }

  // Create new ParallelLoop with conversions to the original induction values.
  // The loop below uses divisions to get the relevant range of values in the
  // new induction value that represent each range of the original induction
  // value. The remainders then determine based on that range, which iteration
  // of the original induction value this represents. This is a normalized value
  // that is un-normalized already by the previous logic.
  auto newPloop = outsideBuilder.create<loop::ParallelOp>(loc, lowerBounds,
                                                          upperBounds, steps);
  OpBuilder insideBuilder(newPloop.region());
  for (unsigned i = 0, e = combinedDimensions.size(); i < e; ++i) {
    Value previous = newPloop.getBody()->getArgument(i);
    unsigned numberCombinedDimensions = combinedDimensions[i].size();
    // Iterate over all except the last induction value.
    for (unsigned j = 0, e = numberCombinedDimensions - 1; j < e; ++j) {
      unsigned idx = combinedDimensions[i][j];

      // Determine the current induction value's current loop iteration
      Value iv = insideBuilder.create<SignedRemIOp>(loc, previous,
                                                    normalizedUpperBounds[idx]);
      replaceAllUsesInRegionWith(loops.getBody()->getArgument(idx), iv,
                                 loops.region());

      // Remove the effect of the current induction value to prepare for the
      // next value.
      previous = insideBuilder.create<SignedDivIOp>(
          loc, previous, normalizedUpperBounds[idx + 1]);
    }

    // The final induction value is just the remaining value.
    unsigned idx = combinedDimensions[i][numberCombinedDimensions - 1];
    replaceAllUsesInRegionWith(loops.getBody()->getArgument(idx), previous,
                               loops.region());
  }

  // Replace the old loop with the new loop.
  loops.getBody()->back().erase();
  newPloop.getBody()->getOperations().splice(
      Block::iterator(newPloop.getBody()->back()),
      loops.getBody()->getOperations());
  loops.erase();
}

void mlir::mapLoopToProcessorIds(loop::ForOp forOp, ArrayRef<Value> processorId,
                                 ArrayRef<Value> numProcessors) {
  assert(processorId.size() == numProcessors.size());
  if (processorId.empty())
    return;

  OpBuilder b(forOp);
  Location loc(forOp.getLoc());
  Value mul = processorId.front();
  for (unsigned i = 1, e = processorId.size(); i < e; ++i)
    mul = b.create<AddIOp>(loc, b.create<MulIOp>(loc, mul, numProcessors[i]),
                           processorId[i]);
  Value lb = b.create<AddIOp>(loc, forOp.lowerBound(),
                              b.create<MulIOp>(loc, forOp.step(), mul));
  forOp.setLowerBound(lb);

  Value step = forOp.step();
  for (auto numProcs : numProcessors)
    step = b.create<MulIOp>(loc, step, numProcs);
  forOp.setStep(step);
}

/// Returns true if the load/store associated with `acc' can be hoisted out of
/// `forOp' (without considering hoisting of the op creating the memref.
static bool isHoistableLoadStore(const MemRefAccess &acc, AffineForOp forOp) {
  Value memref = acc.memref;

  // If the memref is defined in the same for op, can't hoist.
  if (memref.getDefiningOp() &&
      memref.getDefiningOp()->getBlock() == forOp.getBody())
    return false;

  AffineValueMap vmap;
  acc.getAccessMap(&vmap);
  // Check if the access is invariant with respect to this forOp.
  return llvm::find(vmap.getOperands(), forOp.getInductionVar()) ==
         vmap.getOperands().end();
}

/// Returns true if no other affine for op's are nested within.
static bool isInnermostAffineForOp(AffineForOp forOp) {
  // Only for the innermost affine.for op's.
  bool isInnermost = true;
  forOp.walk([&](AffineForOp thisForOp) {
    isInnermost = (thisForOp == forOp);
    return WalkResult::interrupt();
  });
  return isInnermost;
}

/// Returns true if the two memref access provided can't be determined to be
/// either equivalent to and can't be determined to be distinct from each other
/// at compile time; false otherwise. Note that accesses are compared post full
/// composition - so all information up until provenance is captured.
//  Ex: %A[%i][%j], %A[%i][%j] will return true.
//      %A[%i][%j], %A[%i + 1][%j] will return true (since it's known they are
//                                                  different)
//      %A[symbol(%M)], %A[symbol(%M)] will return true.
//      %A[%i][%j], %A[%j][%i] will return false.
//      %A[%M], %A[%N] will return false.
static bool mayBeEqual(const MemRefAccess &A, const MemRefAccess &B) {
  if (A.memref != B.memref)
    return false;

  AffineValueMap diff, AMap, BMap;
  A.getAccessMap(&AMap);
  B.getAccessMap(&BMap);

  AffineValueMap::difference(AMap, BMap, &diff);
  // return !diff.getAffineMap().getResult(0).isa<AffineConstantExpr>();
  return llvm::any_of(diff.getAffineMap().getResults(), [](AffineExpr e) {
    return !e.isa<AffineConstantExpr>();
  });
}

// TODO: only works on innermost loops.
// TODO: does not check for escaping memrefs.
// TODO: only hoists one loop up when it does.
void mlir::scalarReplace(AffineForOp forOp) {
  FuncOp f = forOp.getOperation()->getParentOfType<FuncOp>();
  // Only innermost loops for now.
  if (!isInnermostAffineForOp(forOp))
    return;

  // Constant zero index to avoid duplicates.
  OpBuilder topBuilder(f.getBody());
  Value zeroIndex = topBuilder.create<ConstantIndexOp>(f.getLoc(), 0);

  // Create groups of affine accesses such that each group of affine accesses
  // all refers to the same memref location. It is not feasible to construct a
  // key, but one can check if two affine references access the same element
  // (for a given value of all outer IVs and parameters).
  // TODO: this can be optimized using a disjoint set data structure (union
  // find) if needed.
  std::vector<SmallVector<MemRefAccess, 4>> accessSets;

  LLVM_DEBUG(llvm::dbgs() << "COLLECTING ACCESS SETS\n";);
  // Process all affine load and store ops.
  forOp.walk([&](Operation *op) {
    if (!isa<AffineLoadOp>(op) && !isa<AffineStoreOp>(op))
      return;

    MemRefAccess acc(op);

    // Check if a group of equivalent accesses already exists.
    const auto &en =
        std::find_if(accessSets.begin(), accessSets.end(),
                     [&](const SmallVector<MemRefAccess, 4> &accList) {
                       assert(!accList.empty() && "expected non-empty");
                       return (accList.front() == acc);
                     });
    if (en != accessSets.end()) {
      // If the reference exists, add operation to that group.
      en->push_back(acc);
    } else {
      // Create a new group otherwise.
      accessSets.emplace_back(SmallVector<MemRefAccess, 4>{acc});
    }
  });

  LLVM_DEBUG(llvm::dbgs() << accessSets.size()
                          << " ACCESS SETS TO ITERATE THROUGH\n");

  // Determine which groups are replacable by scalars. Iterate through the
  // disjoint sets of memory accesses.
  std::vector<bool> isScalarReplacable(accessSets.size(), false);
  for (auto &en : llvm::enumerate(accessSets)) {
    const auto &eqAccesses = en.value();
    unsigned i = en.index();

    // Find the first appearing op - the one that dominates everything else in
    // the group: this is the part that needs to be extended to handle
    // non-innermost loops since isBeforeInBlock can longer be used (instead,
    // srcAppearsBeforeDstInCommonBlock is needed).
    assert(!eqAccesses.empty() && "equivalence class can't be empty");
    auto sampleMemOp = eqAccesses.front();

    MemRefAccess acc(sampleMemOp);

    bool containsStore = llvm::any_of(eqAccesses, [](const MemRefAccess &acc) {
      return isa<AffineStoreOp>(acc.opInst);
    });

    if (!containsStore) {
      // All subsequent loads can be replaced with the result of the first
      // load, if stores in all other groups are provably distinct.
      // Check if any of the other groups have a may conflict store.
      if (llvm::any_of(
              accessSets, [&](const SmallVector<MemRefAccess, 4> &accSet) {
                if (llvm::all_of(accSet, [](const MemRefAccess &thisAcc) {
                      return !isa<AffineStoreOp>(thisAcc.opInst);
                    }))
                  // None of them is a store op.
                  return false;
                return mayBeEqual(accSet.front(), acc);
              })) {
        continue;
      }
      isScalarReplacable[i] = true;
      continue;
    }

    // One of the ops is a store.
    // A replacement can only be performed if the memory op's in other group
    // are known to be distinct from this.
    if (llvm::any_of(accessSets,
                     [&](const SmallVector<MemRefAccess, 4> &accList) {
                       return mayBeEqual(acc, accList.front());
                     })) {
      continue;
    }

    // If one of the op's is a store, we will only do the replacement if the
    // accesses are hoistable, and the replacement will be performed using a
    // single element memref.

    // We don't care about the case that's not hoistable for now, as
    // forwardStoreToLoad already handles this.
    if (isHoistableLoadStore(acc, forOp))
      isScalarReplacable[i] = true;
  }

  LLVM_DEBUG(llvm::dbgs() << accessSets.size() << " ITERATING 1st PHASE END\n");

  // Iterate through the disjoint sets of memory accesses.
  for (auto &en : llvm::enumerate(accessSets)) {
    if (!isScalarReplacable[en.index()])
      continue;

    const auto &eqAccesses = en.value();

    // Find the first appearing op - the one that dominates everything else in
    // the group: this is the part that needs to be extended to handle
    // non-innermost loops since isBeforeInBlock can longer be used (instead,
    // srcAppearsBeforeDstInCommonBlock is needed).
    auto *firstMemOp =
        std::min_element(eqAccesses.begin(), eqAccesses.end(),
                         [](const MemRefAccess &a, const MemRefAccess &b) {
                           return a.opInst->isBeforeInBlock(b.opInst);
                         })
            ->opInst;

    MemRefAccess acc(firstMemOp);
    AffineValueMap vMap;
    acc.getAccessMap(&vMap);

    MemRefType origMemrefType = acc.memref.getType().cast<MemRefType>();

    bool containsStore = llvm::any_of(eqAccesses, [](const MemRefAccess &acc) {
      return isa<AffineStoreOp>(acc.opInst);
    });

    if (!containsStore) {
      // All ops in this equivalence class are loads.
      Value scalar;
      bool hoistable = isHoistableLoadStore(acc, forOp);
      if (hoistable) {
        // Hoist the load; create the new load.
        SmallVector<Value, 4> operands;
        operands.reserve(1 + vMap.getNumOperands());
        operands.push_back(acc.memref);
        operands.append(vMap.getOperands().begin(), vMap.getOperands().end());
        // Insert right before the for op.
        OpBuilder b(forOp.getOperation());
        scalar = b.create<AffineLoadOp>(forOp.getLoc(), vMap.getAffineMap(),
                                        operands);
      } else {
        scalar = cast<AffineLoadOp>(firstMemOp).getResult();
      }
      // Erase and replace all uses of existing load op's with the scalar.
      for (auto it = hoistable ? eqAccesses.begin()
                               : std::next(eqAccesses.begin());
           it != eqAccesses.end();) {
        auto loadOp = cast<AffineLoadOp>((*it++).opInst);
        loadOp.getResult().replaceAllUsesWith(scalar);
        loadOp.erase();
      }
      continue;
    }

    // At least one of the ops is a store.
    // Hoistable - create a single element memref.
    OpBuilder b(forOp.getOperation());
    auto singleEltMemRef = b.create<AllocaOp>(
        forOp.getLoc(),
        MemRefType::get(/*shape=*/{1}, origMemrefType.getElementType()));

    // Load from the memref and store to the scalar (one element memref).
    // %singleEltMemref[0] = %A[...];
    Value scalar = b.create<AffineLoadOp>(
        forOp.getLoc(), acc.memref, vMap.getAffineMap(), vMap.getOperands());
    b.create<AffineStoreOp>(forOp.getLoc(), scalar, singleEltMemRef, zeroIndex);

    // Replace all load/stores of original memref with %singleEltMemref[0].
    SmallVector<AffineExpr, 1> resultExprs = {b.getAffineConstantExpr(0)};
    for (auto &acc : eqAccesses) {
      if (failed(replaceAllMemRefUsesWith(
              acc.memref, singleEltMemRef, acc.opInst, {},
              AffineMap::get(origMemrefType.getRank(), 0, resultExprs), {})))
        assert(false && "unimplemented escaping uses");
    }

    // Create the epilogue that stores from the single elt memref back to
    // the original, and dealloc the former.
    // %A[...] = %singleEltMemRef[0]
    b.setInsertionPoint(forOp.getOperation()->getBlock(),
                        std::next(Block::iterator(forOp.getOperation())));
    scalar = b.create<AffineLoadOp>(forOp.getLoc(), singleEltMemRef, zeroIndex);
    b.create<AffineStoreOp>(forOp.getLoc(), scalar, acc.memref,
                            vMap.getAffineMap(), vMap.getOperands());
    // No need of a dealloc since we are using an alloca.
  }

  LLVM_DEBUG(llvm::dbgs() << "SCAL REP END\n");
}

/// Given a memref region, determine the lowest depth at which transfers can be
/// placed for it, and return the corresponding block, start and end positions
/// in the block for placing incoming (read) and outgoing (write) copies
/// respectively. The lowest depth depends on whether the region being accessed
/// is hoistable with respect to one or more immediately surrounding loops.
static void
findHighestBlockForPlacement(const MemRefRegion &region, Block &block,
                             Block::iterator &begin, Block::iterator &end,
                             Block **copyPlacementBlock,
                             Block::iterator *copyInPlacementStart,
                             Block::iterator *copyOutPlacementStart) {
  const auto *cst = region.getConstraints();
  SmallVector<Value, 4> symbols;
  cst->getIdValues(cst->getNumDimIds(), cst->getNumDimAndSymbolIds(), &symbols);

  SmallVector<AffineForOp, 4> enclosingFors;
  getLoopIVs(*block.begin(), &enclosingFors);
  // Walk up loop parents till we find an IV on which this region is
  // symbolic/variant.
  auto it = enclosingFors.rbegin();
  for (auto e = enclosingFors.rend(); it != e; ++it) {
    // TODO(bondhugula): also need to be checking this for regions symbols that
    // aren't loop IVs, whether we are within their resp. defs' dominance scope.
    if (llvm::is_contained(symbols, it->getInductionVar()))
      break;
  }

  if (it != enclosingFors.rbegin()) {
    auto lastInvariantIV = *std::prev(it);
    *copyInPlacementStart = Block::iterator(lastInvariantIV.getOperation());
    *copyOutPlacementStart = std::next(*copyInPlacementStart);
    *copyPlacementBlock = lastInvariantIV.getOperation()->getBlock();
  } else {
    *copyInPlacementStart = begin;
    *copyOutPlacementStart = end;
    *copyPlacementBlock = &block;
  }
}

// Info comprising stride and number of elements transferred every stride.
struct StrideInfo {
  int64_t stride;
  int64_t numEltPerStride;
};

/// Returns striding information for a copy/transfer of this region with
/// potentially multiple striding levels from outermost to innermost. For an
/// n-dimensional region, there can be at most n-1 levels of striding
/// successively nested.
//  TODO(bondhugula): make this work with non-identity layout maps.
static void getMultiLevelStrides(const MemRefRegion &region,
                                 ArrayRef<int64_t> bufferShape,
                                 SmallVectorImpl<StrideInfo> *strideInfos) {
  if (bufferShape.size() <= 1)
    return;

  int64_t numEltPerStride = 1;
  int64_t stride = 1;
  for (int d = bufferShape.size() - 1; d >= 1; d--) {
    int64_t dimSize = region.memref.getType().cast<MemRefType>().getDimSize(d);
    stride *= dimSize;
    numEltPerStride *= bufferShape[d];
    // A stride is needed only if the region has a shorter extent than the
    // memref along the dimension *and* has an extent greater than one along the
    // next major dimension.
    if (bufferShape[d] < dimSize && bufferShape[d - 1] > 1) {
      strideInfos->push_back({stride, numEltPerStride});
    }
  }
}

/// Generates a point-wise copy from/to `memref' to/from `fastMemRef' and
/// returns the outermost AffineForOp of the copy loop nest. `lbMaps` and
/// `ubMaps` along with `lbOperands` and `ubOperands` hold the lower and upper
/// bound information for the copy loop nest. `fastBufOffsets` contain the
/// expressions to be subtracted out from the respective copy loop iterators in
/// order to index the fast buffer. If `copyOut' is true, generates a copy-out;
/// otherwise a copy-in. Builder `b` should be set to the point the copy nest is
/// inserted.
//
/// The copy-in nest is generated as follows as an example for a 2-d region:
/// for x = ...
///   for y = ...
///     fast_buf[x - offset_x][y - offset_y] = memref[x][y]
///
static AffineForOp
generatePointWiseCopy(Location loc, Value memref, Value fastMemRef,
                      ArrayRef<AffineMap> lbMaps, ArrayRef<Value> lbOperands,
                      ArrayRef<AffineMap> ubMaps, ArrayRef<Value> ubOperands,
                      ArrayRef<AffineExpr> fastBufOffsets, bool isCopyOut,
                      OpBuilder b) {
  assert(llvm::all_of(lbMaps, [&](AffineMap lbMap) {
    return lbMap.getNumInputs() == lbOperands.size();
  }));
  assert(llvm::all_of(ubMaps, [&](AffineMap ubMap) {
    return ubMap.getNumInputs() == ubOperands.size();
  }));

  unsigned rank = memref.getType().cast<MemRefType>().getRank();
  assert(lbMaps.size() == rank && "wrong number of lb maps");
  assert(ubMaps.size() == rank && "wrong number of ub maps");

  SmallVector<Value, 4> memIndices;
  SmallVector<AffineExpr, 4> fastBufExprs;
  SmallVector<Value, 4> fastBufMapOperands;
  AffineForOp copyNestRoot;
  SmallVector<AffineApplyOp, 4> mayBeDeadApplys;
  for (unsigned d = 0; d < rank; ++d) {
    auto forOp = createCanonicalizedAffineForOp(b, loc, lbOperands, lbMaps[d],
                                                ubOperands, ubMaps[d]);
    if (d == 0)
      copyNestRoot = forOp;

    b = forOp.getBodyBuilder();

    auto fastBufOffsetMap =
        AffineMap::get(lbOperands.size(), 0, {fastBufOffsets[d]});
    auto offset = b.create<AffineApplyOp>(loc, fastBufOffsetMap, lbOperands);

    // Construct the subscript for the fast memref being copied into/from:
    // x - offset_x.
    fastBufExprs.push_back(b.getAffineDimExpr(2 * d + 1) -
                           b.getAffineDimExpr(2 * d));
    fastBufMapOperands.push_back(offset);
    fastBufMapOperands.push_back(forOp.getInductionVar());
    mayBeDeadApplys.push_back(offset);

    // Subscript for the slow memref being copied.
    memIndices.push_back(forOp.getInductionVar());
  }

  auto fastBufMap = AffineMap::get(2 * rank, /*symbolCount=*/0, fastBufExprs);
  fullyComposeAffineMapAndOperands(&fastBufMap, &fastBufMapOperands);
  fastBufMap = simplifyAffineMap(fastBufMap);
  canonicalizeMapAndOperands(&fastBufMap, &fastBufMapOperands);

  // Drop any dead affine.applys.
  for (auto applyOp : mayBeDeadApplys)
    if (applyOp.use_empty())
      applyOp.erase();

  if (!isCopyOut) {
    // Copy in.
    auto load = b.create<AffineLoadOp>(loc, memref, memIndices);
    b.create<AffineStoreOp>(loc, load, fastMemRef, fastBufMap,
                            fastBufMapOperands);
    return copyNestRoot;
  }

  // Copy out.
  auto load =
      b.create<AffineLoadOp>(loc, fastMemRef, fastBufMap, fastBufMapOperands);
  b.create<AffineStoreOp>(loc, load, memref, memIndices);
  return copyNestRoot;
}

static InFlightDiagnostic LLVM_ATTRIBUTE_UNUSED
emitRemarkForBlock(Block &block) {
  return block.getParentOp()->emitRemark();
}

/// Creates a buffer in the faster memory space for the specified memref region;
/// generates a copy from the lower memory space to this one, and replaces all
/// loads/stores in the block range [`begin', `end') of `block' to load/store
/// from that buffer. Returns failure if copies could not be generated due to
/// yet unimplemented cases. `copyInPlacementStart` and `copyOutPlacementStart`
/// in copyPlacementBlock specify the insertion points where the incoming copies
/// and outgoing copies, respectively, should be inserted (the insertion happens
/// right before the insertion point). Since `begin` can itself be invalidated
/// due to the memref rewriting done from this method, the output argument
/// `nBegin` is set to its replacement (set to `begin` if no invalidation
/// happens). Since outgoing copies could have  been inserted at `end`, the
/// output argument `nEnd` is set to the new end. `sizeInBytes` is set to the
/// size of the fast buffer allocated.
static LogicalResult generateCopy(
    const MemRefRegion &region, Block *block, Block::iterator begin,
    Block::iterator end, Block *copyPlacementBlock,
    Block::iterator copyInPlacementStart, Block::iterator copyOutPlacementStart,
    AffineCopyOptions copyOptions, DenseMap<Value, Value> &fastBufferMap,
    DenseSet<Operation *> &copyNests, uint64_t *sizeInBytes,
    Block::iterator *nBegin, Block::iterator *nEnd) {
  *nBegin = begin;
  *nEnd = end;

  FuncOp f = begin->getParentOfType<FuncOp>();
  OpBuilder topBuilder(f.getBody());
  Value zeroIndex = topBuilder.create<ConstantIndexOp>(f.getLoc(), 0);

  if (begin == end)
    return success();

  // Is the copy out point at the end of the block where we are doing
  // explicit copying.
  bool isCopyOutAtEndOfBlock = (end == copyOutPlacementStart);

  // Copies for read regions are going to be inserted at 'begin'.
  OpBuilder prologue(copyPlacementBlock, copyInPlacementStart);
  // Copies for write regions are going to be inserted at 'end'.
  OpBuilder epilogue(copyPlacementBlock, copyOutPlacementStart);
  OpBuilder &b = region.isWrite() ? epilogue : prologue;

  // Builder to create constants at the top level.
  auto func = copyPlacementBlock->getParent()->getParentOfType<FuncOp>();
  OpBuilder top(func.getBody());

  auto loc = region.loc;
  auto memref = region.memref;
  auto memRefType = memref.getType().cast<MemRefType>();

  auto layoutMaps = memRefType.getAffineMaps();
  if (layoutMaps.size() > 1 ||
      (layoutMaps.size() == 1 && !layoutMaps[0].isIdentity())) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return failure();
  }

  // Indices to use for the copying.
  // Indices for the original memref being copied from/to.
  SmallVector<Value, 4> memIndices;
  // Indices for the faster buffer being copied into/from.
  SmallVector<Value, 4> bufIndices;

  unsigned rank = memRefType.getRank();
  SmallVector<int64_t, 4> fastBufferShape;
  AffineMap fastBufferLayout = copyOptions.fastBufferLayout
                                   ? copyOptions.fastBufferLayout
                                   : b.getMultiDimIdentityMap(rank);

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  Optional<int64_t> numElements = region.getConstantBoundingSizeAndShape(
      &fastBufferShape, &lbs, &lbDivisors);
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
    return failure();
  }

  if (numElements.getValue() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to copy\n");
    *sizeInBytes = 0;
    return success();
  }

  SmallVector<AffineMap, 4> lbMaps(rank), ubMaps(rank);
  for (unsigned i = 0; i < rank; ++i)
    region.getLowerAndUpperBound(i, lbMaps[i], ubMaps[i]);

  const FlatAffineConstraints *cst = region.getConstraints();
  // 'regionSymbols' hold values that this memory region is symbolic/parametric
  // on; these typically include loop IVs surrounding the level at which the
  // copy generation is being done or other valid symbols in MLIR.
  SmallVector<Value, 8> regionSymbols;
  cst->getIdValues(rank, cst->getNumIds(), &regionSymbols);

  // Construct the index expressions for the fast memory buffer. The index
  // expression for a particular dimension of the fast buffer is obtained by
  // subtracting out the lower bound on the original memref's data region
  // along the corresponding dimension.

  // Index start offsets for faster memory buffer relative to the original.
  SmallVector<AffineExpr, 4> fastBufOffsets;
  fastBufOffsets.reserve(rank);
  for (unsigned d = 0; d < rank; d++) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++)
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);

    // Set copy start location for this dimension in the lower memory space
    // memref.
    if (auto caf = offset.dyn_cast<AffineConstantExpr>()) {
      auto indexVal = caf.getValue();
      if (indexVal == 0) {
        memIndices.push_back(zeroIndex);
      } else {
        memIndices.push_back(
            top.create<ConstantIndexOp>(loc, indexVal).getResult());
      }
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      auto map = AffineMap::get(
          cst->getNumDimIds() + cst->getNumSymbolIds() - rank, 0, offset);
      memIndices.push_back(b.create<AffineApplyOp>(loc, map, regionSymbols));
    }
    // The fast buffer is copied into at location zero; addressing is relative.
    bufIndices.push_back(zeroIndex);

    // Record the offsets since they are needed to remap the memory accesses of
    // the original memref further below.
    fastBufOffsets.push_back(offset);
  }

  // The faster memory space buffer.
  Value fastMemRef;

  // Check if a buffer was already created.
  bool existingBuf = fastBufferMap.count(memref) > 0;
  if (!existingBuf) {
    auto fastMemRefType =
        MemRefType::get(fastBufferShape, memRefType.getElementType(),
                        fastBufferLayout, copyOptions.fastMemorySpace);

    // Create the fast memory space buffer just before the 'affine.for'
    // operation.
    fastMemRef = prologue.create<AllocOp>(loc, fastMemRefType).getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    *sizeInBytes = getMemRefSizeInBytes(fastMemRefType).getValue();
    LLVM_DEBUG(emitRemarkForBlock(*block)
               << "Creating fast buffer of type " << fastMemRefType
               << " and size " << llvm::divideCeil(*sizeInBytes, 1024)
               << " KiB\n");
  } else {
    // Reuse the one already created.
    fastMemRef = fastBufferMap[memref];
    *sizeInBytes = 0;
  }

  auto numElementsSSA =
      top.create<ConstantIndexOp>(loc, numElements.getValue());

  Value dmaStride = nullptr;
  Value numEltPerDmaStride = nullptr;
  if (copyOptions.generateDma) {
    SmallVector<StrideInfo, 4> dmaStrideInfos;
    getMultiLevelStrides(region, fastBufferShape, &dmaStrideInfos);

    // TODO(bondhugula): use all stride levels once DmaStartOp is extended for
    // multi-level strides.
    if (dmaStrideInfos.size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "Only up to one level of stride supported\n");
      return failure();
    }

    if (!dmaStrideInfos.empty()) {
      dmaStride = top.create<ConstantIndexOp>(loc, dmaStrideInfos[0].stride);
      numEltPerDmaStride =
          top.create<ConstantIndexOp>(loc, dmaStrideInfos[0].numEltPerStride);
    }
  }

  // Record the last operation where we want the memref replacement to end. We
  // later do the memref replacement only in [begin, postDomFilter] so
  // that the original memref's used in the data movement code themselves don't
  // get replaced.
  auto postDomFilter = std::prev(end);

  // Create fully composed affine maps for each memref.
  auto memAffineMap = b.getMultiDimIdentityMap(memIndices.size());
  fullyComposeAffineMapAndOperands(&memAffineMap, &memIndices);
  auto bufAffineMap = b.getMultiDimIdentityMap(bufIndices.size());
  fullyComposeAffineMapAndOperands(&bufAffineMap, &bufIndices);

  if (!copyOptions.generateDma) {
    // Point-wise copy generation.
    auto copyNest =
        generatePointWiseCopy(loc, memref, fastMemRef, lbMaps,
                              /*lbOperands=*/regionSymbols, ubMaps,
                              /*ubOperands=*/regionSymbols, fastBufOffsets,
                              /*isCopyOut=*/region.isWrite(), b);

    // Record this so that we can skip it from yet another copy.
    copyNests.insert(copyNest);

    // Since new ops are being appended (for copy out's), adjust the end to
    // mark end of block range being processed if necessary.
    if (region.isWrite() && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(copyNest.getOperation());
  } else {
    // DMA generation.
    // Create a tag (single element 1-d memref) for the DMA.
    auto tagMemRefType = MemRefType::get({1}, top.getIntegerType(32), {},
                                         copyOptions.tagMemorySpace);
    auto tagMemRef = prologue.create<AllocOp>(loc, tagMemRefType);

    SmallVector<Value, 4> tagIndices({zeroIndex});
    auto tagAffineMap = b.getMultiDimIdentityMap(tagIndices.size());
    fullyComposeAffineMapAndOperands(&tagAffineMap, &tagIndices);
    if (!region.isWrite()) {
      // DMA non-blocking read from original buffer to fast buffer.
      b.create<AffineDmaStartOp>(loc, memref, memAffineMap, memIndices,
                                 fastMemRef, bufAffineMap, bufIndices,
                                 tagMemRef, tagAffineMap, tagIndices,
                                 numElementsSSA, dmaStride, numEltPerDmaStride);
    } else {
      // DMA non-blocking write from fast buffer to the original memref.
      auto op = b.create<AffineDmaStartOp>(
          loc, fastMemRef, bufAffineMap, bufIndices, memref, memAffineMap,
          memIndices, tagMemRef, tagAffineMap, tagIndices, numElementsSSA,
          dmaStride, numEltPerDmaStride);
      // Since new ops may be appended at 'end' (for outgoing DMAs), adjust the
      // end to mark end of block range being processed.
      if (isCopyOutAtEndOfBlock)
        *nEnd = Block::iterator(op.getOperation());
    }

    // Matching DMA wait to block on completion; tag always has a 0 index.
    b.create<AffineDmaWaitOp>(loc, tagMemRef, tagAffineMap, zeroIndex,
                              numElementsSSA);

    // Generate dealloc for the tag.
    auto tagDeallocOp = epilogue.create<DeallocOp>(loc, tagMemRef);
    if (*nEnd == end && isCopyOutAtEndOfBlock)
      // Since new ops are being appended (for outgoing DMAs), adjust the end to
      // mark end of range of the original.
      *nEnd = Block::iterator(tagDeallocOp.getOperation());
  }

  // Generate dealloc for the buffer.
  if (!existingBuf) {
    auto bufDeallocOp = epilogue.create<DeallocOp>(loc, fastMemRef);
    // When generating pointwise copies, `nEnd' has to be set to deallocOp on
    // the fast buffer (since it marks the new end insertion point).
    if (!copyOptions.generateDma && *nEnd == end && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(bufDeallocOp.getOperation());
  }

  // Replace all uses of the old memref with the faster one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  // Ex: to replace load %A[%i, %j] with load %Abuf[%i - %iT, %j - %jT],
  // index remap will be (%i, %j) -> (%i - %iT, %j - %jT),
  // i.e., affine.apply (d0, d1, d2, d3) -> (d2-d0, d3-d1) (%iT, %jT, %i, %j),
  // and (%iT, %jT) will be the 'extraOperands' for 'rep all memref uses with'.
  // d2, d3 correspond to the original indices (%i, %j).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    // The starting operands of indexRemap will be regionSymbols (the symbols on
    // which the memref region is parametric); then those corresponding to
    // the memref's original indices follow.
    auto dimExpr = b.getAffineDimExpr(regionSymbols.size() + i);
    remapExprs.push_back(dimExpr - fastBufOffsets[i]);
  }
  auto indexRemap = AffineMap::get(regionSymbols.size() + rank, 0, remapExprs);

  // Record the begin since it may be invalidated by memref replacement.
  Block::iterator prevOfBegin;
  bool isBeginAtStartOfBlock = (begin == block->begin());
  if (!isBeginAtStartOfBlock)
    prevOfBegin = std::prev(begin);

  // *Only* those uses within the range [begin, end) of 'block' are replaced.
  replaceAllMemRefUsesWith(memref, fastMemRef,
                           /*extraIndices=*/{}, indexRemap,
                           /*extraOperands=*/regionSymbols,
                           /*symbolOperands=*/{},
                           /*domInstFilter=*/&*begin,
                           /*postDomInstFilter=*/&*postDomFilter);

  *nBegin = isBeginAtStartOfBlock ? block->begin() : std::next(prevOfBegin);

  return success();
}

/// Construct the memref region to just include the entire memref. Returns false
/// dynamic shaped memref's for now. `numParamLoopIVs` is the number of
/// enclosing loop IVs of `op` (starting from the outermost) that the region
/// is parametric on.
static bool getFullMemRefAsRegion(Operation *op, unsigned numParamLoopIVs,
                                  MemRefRegion *region) {
  unsigned rank;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    rank = loadOp.getMemRefType().getRank();
    region->memref = loadOp.getMemRef();
    region->setWrite(false);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    rank = storeOp.getMemRefType().getRank();
    region->memref = storeOp.getMemRef();
    region->setWrite(true);
  } else {
    assert(false && "expected load or store op");
    return false;
  }
  auto memRefType = region->memref.getType().cast<MemRefType>();
  if (!memRefType.hasStaticShape())
    return false;

  auto *regionCst = region->getConstraints();

  // Just get the first numSymbols IVs, which the memref region is parametric
  // on.
  SmallVector<AffineForOp, 4> ivs;
  getLoopIVs(*op, &ivs);
  ivs.resize(numParamLoopIVs);
  SmallVector<Value, 4> symbols;
  extractForInductionVars(ivs, &symbols);
  regionCst->reset(rank, numParamLoopIVs, 0);
  regionCst->setIdValues(rank, rank + numParamLoopIVs, symbols);

  // Memref dim sizes provide the bounds.
  for (unsigned d = 0; d < rank; d++) {
    auto dimSize = memRefType.getDimSize(d);
    assert(dimSize > 0 && "filtered dynamic shapes above");
    regionCst->addConstantLowerBound(d, 0);
    regionCst->addConstantUpperBound(d, dimSize - 1);
  }
  return true;
}

/// Performs explicit copying for the contiguous sequence of operations in the
/// block iterator range [`begin', `end'), where `end' can't be past the
/// terminator of the block (since additional operations are potentially
/// inserted right before `end`. Returns the total size of fast memory space
/// buffers used. `copyOptions` provides various parameters, and the output
/// argument `copyNests` is the set of all copy nests inserted, each represented
/// by its root affine.for. Since we generate alloc's and dealloc's for all fast
/// buffers (before and after the range of operations resp. or at a hoisted
/// position), all of the fast memory capacity is assumed to be available for
/// processing this block range. When 'filterMemRef' is specified, copies are
/// only generated for the provided MemRef.
uint64_t mlir::affineDataCopyGenerate(Block::iterator begin,
                                      Block::iterator end,
                                      const AffineCopyOptions &copyOptions,
                                      Optional<Value> filterMemRef,
                                      DenseSet<Operation *> &copyNests,
                                      SmallVectorImpl<Value> *fastBufs) {
  if (begin == end)
    return 0;

  assert(begin->getBlock() == std::prev(end)->getBlock() &&
         "Inconsistent block begin/end args");
  assert(end != end->getBlock()->end() && "end can't be the block terminator");

  Block *block = begin->getBlock();

  // Copies will be generated for this depth, i.e., symbolic in all loops
  // surrounding the this block range.
  unsigned copyDepth = getNestingDepth(&*begin);

  LLVM_DEBUG(llvm::dbgs() << "Generating copies at depth " << copyDepth
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "from begin: " << *begin << "\n");
  LLVM_DEBUG(llvm::dbgs() << "to inclusive end: " << *std::prev(end) << "\n");

  // List of memory regions to copy for. We need a map vector to have a
  // guaranteed iteration order to write test cases. CHECK-DAG doesn't help here
  // since the alloc's for example are identical except for the SSA id.
  SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4> readRegions;
  SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4> writeRegions;

  // Map from original memref's to the fast buffers that their accesses are
  // replaced with.
  DenseMap<Value, Value> fastBufferMap;

  // To check for errors when walking the block.
  bool error = false;

  // Walk this range of operations  to gather all memory regions.
  block->walk(begin, end, [&](Operation *opInst) {
    // Gather regions to allocate to buffers in faster memory space.
    if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
      if ((filterMemRef.hasValue() && filterMemRef != loadOp.getMemRef()) ||
          (loadOp.getMemRefType().getMemorySpace() !=
           copyOptions.slowMemorySpace))
        return;
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
      if ((filterMemRef.hasValue() && filterMemRef != storeOp.getMemRef()) ||
          storeOp.getMemRefType().getMemorySpace() !=
              copyOptions.slowMemorySpace)
        return;
    } else {
      // Neither load nor a store op.
      return;
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(region->compute(opInst, copyDepth, /*sliceState=*/nullptr,
                               /*addMemRefDimBounds=*/false))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Error obtaining memory region: semi-affine maps?\n");
      LLVM_DEBUG(llvm::dbgs() << "over-approximating to the entire memref\n");
      if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
        LLVM_DEBUG(
            opInst->emitError("non-constant memref sizes not yet supported"));
        error = true;
        return;
      }
    }

    // Each memref has a single buffer associated with it irrespective of how
    // many load's and store's happen on it.
    // TODO(bondhugula): in the future, when regions don't intersect and satisfy
    // other properties (based on load/store regions), we could consider
    // multiple buffers per memref.

    // Add to the appropriate region if it's not already in it, or take a
    // bounding box union with the existing one if it's already in there.
    // Note that a memref may have both read and write regions - so update the
    // region in the other list if one exists (write in case of read and vice
    // versa) since there is a single bounding box for a memref across all reads
    // and writes that happen on it.

    // Attempts to update; returns true if 'region' exists in targetRegions.
    auto updateRegion =
        [&](const SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4>
                &targetRegions) {
          auto it = targetRegions.find(region->memref);
          if (it == targetRegions.end())
            return false;

          // Perform a union with the existing region.
          if (failed(it->second->unionBoundingBox(*region))) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Memory region bounding box failed; "
                          "over-approximating to the entire memref\n");
            // If the union fails, we will overapproximate.
            if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
              LLVM_DEBUG(opInst->emitError(
                  "non-constant memref sizes not yet supported"));
              error = true;
              return true;
            }
            it->second->getConstraints()->clearAndCopyFrom(
                *region->getConstraints());
          } else {
            // Union was computed and stored in 'it->second': copy to 'region'.
            region->getConstraints()->clearAndCopyFrom(
                *it->second->getConstraints());
          }
          return true;
        };

    bool existsInRead = updateRegion(readRegions);
    if (error)
      return;
    bool existsInWrite = updateRegion(writeRegions);
    if (error)
      return;

    // Finally add it to the region list.
    if (region->isWrite() && !existsInWrite) {
      writeRegions[region->memref] = std::move(region);
    } else if (!region->isWrite() && !existsInRead) {
      readRegions[region->memref] = std::move(region);
    }
  });

  if (error) {
    begin->emitError(
        "copy generation failed for one or more memref's in this block\n");
    return 0;
  }

  uint64_t totalCopyBuffersSizeInBytes = 0;
  bool ret = true;
  auto processRegions =
      [&](const SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4>
              &regions) {
        for (const auto &regionEntry : regions) {
          // For each region, hoist copy in/out past all hoistable
          // 'affine.for's.
          Block::iterator copyInPlacementStart, copyOutPlacementStart;
          Block *copyPlacementBlock;
          findHighestBlockForPlacement(
              *regionEntry.second, *block, begin, end, &copyPlacementBlock,
              &copyInPlacementStart, &copyOutPlacementStart);

          uint64_t sizeInBytes;
          Block::iterator nBegin, nEnd;
          LogicalResult iRet = generateCopy(
              *regionEntry.second, block, begin, end, copyPlacementBlock,
              copyInPlacementStart, copyOutPlacementStart, copyOptions,
              fastBufferMap, copyNests, &sizeInBytes, &nBegin, &nEnd);
          if (succeeded(iRet)) {
            // begin/end could have been invalidated, and need update.
            begin = nBegin;
            end = nEnd;
            totalCopyBuffersSizeInBytes += sizeInBytes;
          }
          ret = ret & succeeded(iRet);
        }
      };
  processRegions(readRegions);
  processRegions(writeRegions);

  if (!ret) {
    begin->emitError(
        "copy generation failed for one or more memref's in this block\n");
    return totalCopyBuffersSizeInBytes;
  }

  // For a range of operations, a note will be emitted at the caller.
  AffineForOp forOp;
  uint64_t sizeInKib = llvm::divideCeil(totalCopyBuffersSizeInBytes, 1024);
  if (llvm::DebugFlag && (forOp = dyn_cast<AffineForOp>(&*begin))) {
    forOp.emitRemark()
        << sizeInKib
        << " KiB of copy buffers in fast memory space for this block\n";
  }

  if (totalCopyBuffersSizeInBytes > copyOptions.fastMemCapacityBytes) {
    StringRef str = "Total size of all copy buffers' for this block "
                    "exceeds fast memory capacity\n";
    block->getParentOp()->emitWarning(str);
  }

  if (fastBufs) {
    fastBufs->clear();
    fastBufs->reserve(fastBufferMap.size());
    for (const auto &entry : fastBufferMap) {
      fastBufs->push_back(entry.second);
    }
  }

  return totalCopyBuffersSizeInBytes;
}

// A convenience version of affineDataCopyGenerate for all ops in the body of
// an AffineForOp.
uint64_t mlir::affineDataCopyGenerate(AffineForOp forOp,
                                      const AffineCopyOptions &copyOptions,
                                      Optional<Value> filterMemRef,
                                      DenseSet<Operation *> &copyNests,
                                      SmallVectorImpl<Value> *fastBufs) {
  return affineDataCopyGenerate(forOp.getBody()->begin(),
                                std::prev(forOp.getBody()->end()), copyOptions,
                                filterMemRef, copyNests, fastBufs);
}

/// Returns scalars other than other those of index type that are live in to
/// 'forOp'.
static void getNonIndexLiveInScalars(AffineForOp forOp,
                                     SmallVectorImpl<Value> &scalars) {
  SmallVector<AffineForOp, 4> ivs;
  forOp.walk([&](Operation *op) {
    for (auto value : op->getOperands()) {
      auto type = value.getType();
      if (type.isa<MemRefType>() || type.isa<IndexType>())
        continue;
      if (auto *defOp = value.getDefiningOp()) {
        ivs.clear();
        // Check whether the defining op is outside iv.
        getLoopIVs(*defOp, &ivs);
        if (llvm::find(ivs, forOp) == ivs.end())
          scalars.push_back(value);
      } else {
        scalars.push_back(value);
      }
    }
  });
}

/// Given an input type, provides a vector type for it of the provided width.
static VectorType getVectorizedType(Type inputType, unsigned width) {
  assert(width > 1 && "unexpected vector width");
  assert(!inputType.isa<IndexType>() && "index type can't be vectorized");
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
static Value createVectorMemRef(Value scalMemRef, unsigned vectorWidth) {
  auto scalMemRefType = scalMemRef.getType().cast<MemRefType>();
  auto shape = scalMemRefType.getShape();

  OpBuilder b(scalMemRef.getContext());
  if (auto *defOp = scalMemRef.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  else
    b.setInsertionPointToStart(scalMemRef.cast<BlockArgument>().getOwner());

  auto vecMemRefEltType =
      getVectorizedType(scalMemRefType.getElementType(), vectorWidth);

  SmallVector<int64_t, 4> vecMemRefShape(shape.begin(), shape.end());
  if (vecMemRefShape.back() != -1)
    vecMemRefShape.back() /= vectorWidth;

  auto vecMemRefType = MemRefType::get(vecMemRefShape, vecMemRefEltType);

  // FIXME: we are using a shape cast here, but we do not know whether the base
  // memref is aligned to the right boundary. The load/stores on cast memref (of
  // vector elt type) would be mapped to aligned load/stores by default and
  // lead to a protection fault.
  // We are going to fix this at least where we have access to the defining
  // alloc op.
  if (auto allocOp = dyn_cast_or_null<AllocOp>(scalMemRef.getDefiningOp()))
    allocOp.alignmentAttr(
        b.getI64IntegerAttr(vecMemRefEltType.getSizeInBits() / 8));

  return b.create<MemRefShapeCastOp>(b.getUnknownLoc(), vecMemRefType,
                                     scalMemRef);
}

/// Returns an affine map with the last result of `input' scaled down by
/// `factor'.
static AffineMap scaleDownLastResult(AffineMap input, int64_t factor) {
  SmallVector<AffineExpr, 4> results(input.getResults().begin(),
                                     input.getResults().end());
  results.back() = results.back().floorDiv(factor);
  return AffineMap::get(input.getNumDims(), input.getNumSymbols(), results);
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

  LLVM_DEBUG(llvm::dbgs() << "Vectorizing leaf op " << *op << "\n");

  SmallVector<Type, 8> vectorTypes;
  for (auto v : op->getResults())
    vectorTypes.push_back(getVectorizedType(v.getType(), width));

  // Check whether any operand is null; if so, vectorization failed.
  bool success = llvm::all_of(
      op->getOperands(), [](Value v) { return v.getType().isa<VectorType>(); });
  if (!success) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n[affine-vect]+++++ operands should've been vectorized\n");
    return nullptr;
  }

  OpBuilder b(op);
  OperationState newOp(op->getLoc(), op->getName().getStringRef(),
                       op->getOperands(), vectorTypes, op->getAttrs(),
                       /*successors=*/{},
                       /*regions=*/{}, op->hasResizableOperandsList());
  return b.createOperation(newOp);
}

LogicalResult mlir::loopVectorize(AffineForOp forOp, unsigned simdWidth,
                                  DenseMap<Value, Value> *vecMemRefMap) {
  LLVM_DEBUG(llvm::dbgs() << "Vectorizing " << *forOp << "\n");

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
  DenseMap<Value, Value> toVecMemRefMap;
  SetVector<Value> toVecMemRefs;

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

    Value memref = loadOp ? loadOp.getMemRef() : storeOp.getMemRef();

    if (loadOp)
      toVecLoadOps.insert(loadOp);
    else
      toVecStoreOps.insert(storeOp);

    if (toVecMemRefs.count(memref) == 0)
      toVecMemRefs.insert(memref);

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
  for (auto memref : toVecMemRefs) {
    auto memrefType = memref.getType().cast<MemRefType>();
    auto eltType = memrefType.getElementType();
    if (eltType.isa<VectorType>()) {
      LLVM_DEBUG(llvm::dbgs() << "code already vectorized?\n");
      return failure();
    }

    if (simdWidth % eltType.getIntOrFloatBitWidth() != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "scalar width does not divide h/w vector width\n");
      return failure();
    }
    unsigned thisVectorWidth = simdWidth / eltType.getIntOrFloatBitWidth();
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

  // Check if all live-in scalars are of non-memref/vector/tensor/index type
  // since we can't splat these. Index types are use for subscript computations
  // or loop bound calculations, and aren't supported as operands of operations
  // that need to be vectorized.
  SmallVector<Value, 4> liveInScalars;
  getNonIndexLiveInScalars(forOp, liveInScalars);
  if (llvm::any_of(liveInScalars, [](Value v) {
        auto type = v.getType();
        return type.isa<VectorType>() || type.isa<TensorType>() ||
               type.isa<IndexType>();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Non-scalar type live in - can't splat\n");
    return failure();
  }

  // FIXME: what is the assumption on layouts maps?

  // Create vector memrefs for the ones that will have their load/stores
  // vectorized.
  for (auto vecMemRef : toVecMemRefs) {
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
    auto vecLoadOp = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), toVecMemRefMap[loadOp.getMemRef()],
        scaleDownLastResult(loadOp.getAffineMap(), vectorWidth),
        loadOp.getMapOperands());
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
    rewriter.create<AffineStoreOp>(
        storeOp.getLoc(), storeOp.getValueToStore(),
        toVecMemRefMap[storeOp.getMemRef()],
        scaleDownLastResult(storeOp.getAffineMap(), vectorWidth),
        storeOp.getMapOperands());
    storeOp.erase();
  }

  // Splat live-in scalars.
  for (auto scalar : liveInScalars) {
    OpBuilder rewriter(scalar.getContext());
    Location loc = rewriter.getUnknownLoc();
    if (auto *defOp = scalar.getDefiningOp()) {
      loc = defOp->getLoc();
      rewriter.setInsertionPointAfter(defOp);
    } else {
      auto *block = scalar.cast<BlockArgument>().getOwner();
      loc = block->getParentOp()->getLoc();
      rewriter.setInsertionPointToStart(block);
    }
    auto splat = rewriter.create<SplatOp>(
        loc, scalar, getVectorizedType(scalar.getType(), vectorWidth));
    replaceAllUsesInRegionWith(scalar, splat, forOp.region());
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
  applyPatternsAndFoldGreedily(forOp.getParentOfType<FuncOp>(),
                               std::move(patterns));

  if (vecMemRefMap)
    *vecMemRefMap = std::move(toVecMemRefMap);

  return success();
}

LogicalResult mlir::generateCopyForMemRegion(
    const MemRefRegion &memrefRegion, Operation *analyzedOp,
    const AffineCopyOptions &copyOptions, CopyGenerateResult &result) {
  Block *block = analyzedOp->getBlock();
  auto begin = analyzedOp->getIterator();
  auto end = std::next(begin);
  DenseMap<Value, Value> fastBufferMap;
  DenseSet<Operation *> copyNests;

  auto err = generateCopy(memrefRegion, block, begin, end, block, begin, end,
                          copyOptions, fastBufferMap, copyNests,
                          &result.sizeInBytes, &begin, &end);
  if (failed(err))
    return err;

  result.alloc =
      fastBufferMap.find(memrefRegion.memref)->second.getDefiningOp();
  assert(copyNests.size() <= 1 && "At most one copy nest is expected.");
  result.copyNest = copyNests.empty() ? nullptr : *copyNests.begin();
  return success();
}

/// Gathers all AffineForOps in 'block' at 'currLoopDepth' in 'depthToLoops'.
static void
gatherLoopsInBlock(Block *block, unsigned currLoopDepth,
                   std::vector<SmallVector<AffineForOp, 2>> &depthToLoops) {
  // Add a new empty level to output if it doesn't exist level already.
  assert(currLoopDepth <= depthToLoops.size() && "Unexpected currLoopDepth");
  if (currLoopDepth == depthToLoops.size())
    depthToLoops.push_back(SmallVector<AffineForOp, 2>());

  for (auto &op : *block) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      depthToLoops[currLoopDepth].push_back(forOp);
      gatherLoopsInBlock(forOp.getBody(), currLoopDepth + 1, depthToLoops);
    }
  }
}

/// Gathers all AffineForOps in 'func' grouped by loop depth.
void mlir::gatherLoops(FuncOp func,
                       std::vector<SmallVector<AffineForOp, 2>> &depthToLoops) {
  for (auto &block : func)
    gatherLoopsInBlock(&block, /*currLoopDepth=*/0, depthToLoops);

  // Remove last loop level from output since it's empty.
  if (!depthToLoops.empty()) {
    assert(depthToLoops.back().empty() && "Last loop level is not empty?");
    depthToLoops.pop_back();
  }
}

// TODO: if necessary, this can be extended to also compose in any
// affine.applys, fold to constant if all result dimensions of the map are
// constant (canonicalizeMapAndOperands below already does this for single
// result bound maps), and use simplifyMap to perform algebraic simplification.
AffineForOp mlir::createCanonicalizedAffineForOp(
    OpBuilder b, Location loc, ValueRange lbOperands, AffineMap lbMap,
    ValueRange ubOperands, AffineMap ubMap, int64_t step) {
  SmallVector<Value, 4> lowerOperands(lbOperands);
  SmallVector<Value, 4> upperOperands(ubOperands);

  fullyComposeAffineMapAndOperands(&lbMap, &lowerOperands);
  canonicalizeMapAndOperands(&lbMap, &lowerOperands);
  lbMap = removeDuplicateExprs(lbMap);
  fullyComposeAffineMapAndOperands(&ubMap, &upperOperands);
  canonicalizeMapAndOperands(&ubMap, &upperOperands);
  ubMap = removeDuplicateExprs(ubMap);

  return b.create<AffineForOp>(loc, lowerOperands, lbMap, upperOperands, ubMap,
                               step);
}

/// Creates an AffineIfOp that encodes the conditional to choose between
/// the constant trip count version and an unknown trip count version of this
/// nest of loops. This is used to separate partial and full tiles if `loops`
/// has the intra-tile loops. The affine.if op is inserted at the builder
/// insertion point of `b`.
static AffineIfOp createSeparationCondition(MutableArrayRef<AffineForOp> loops,
                                            OpBuilder b) {
  if (loops.empty())
    return nullptr;

  auto *context = loops[0].getContext();

  FlatAffineConstraints cst;
  getIndexSet(loops, &cst);

  // Remove constraints that are independent of these loop IVs.
  cst.removeIndependentConstraints(/*pos=*/0, /*num=*/loops.size());

  // Construct the constraint set representing the guard for full tiles. The
  // lower bound (and upper bound) corresponding to the full tile should be
  // larger (and resp. smaller) than any other lower (or upper bound).
  SmallVector<int64_t, 8> fullTileLb, fullTileUb;
  for (auto loop : loops) {
    (void)loop;
    // TODO: Non-unit stride is not an issue to generalize to.
    assert(loop.getStep() == 1 && "point loop step expected to be one");
    // Mark everything symbols for the purpose of finding a constant diff pair.
    cst.setDimSymbolSeparation(/*newSymbolCount=*/cst.getNumDimAndSymbolIds() -
                               1);
    unsigned fullTileLbPos, fullTileUbPos;
    if (!cst.getConstantBoundOnDimSize(0, /*lb=*/nullptr,
                                       /*lbFloorDivisor=*/nullptr,
                                       /*ub=*/nullptr, &fullTileLbPos,
                                       &fullTileUbPos)) {
      LLVM_DEBUG(llvm::dbgs() << "Can't get constant diff pair for a loop\n");
      return nullptr;
    }

    SmallVector<unsigned, 4> lbIndices, ubIndices;
    cst.getLowerAndUpperBoundIndices(/*pos=*/0, &lbIndices, &ubIndices);

    auto fLb = cst.getInequality(fullTileLbPos);
    auto fUb = cst.getInequality(fullTileUbPos);
    fullTileLb.assign(fLb.begin(), fLb.end());
    fullTileUb.assign(fUb.begin(), fUb.end());

    // Full tile lower bound should be >= than any other lower bound.
    for (auto lbIndex : lbIndices)
      for (unsigned i = 0, e = cst.getNumCols(); i < e; ++i)
        cst.atIneq(lbIndex, i) = fullTileLb[i] - cst.atIneq(lbIndex, i);

    // Full tile upper bound should be <= any other upper bound.
    for (auto ubIndex : ubIndices)
      for (unsigned i = 0, e = cst.getNumCols(); i < e; ++i)
        cst.atIneq(ubIndex, i) -= fullTileUb[i];

    cst.removeId(0);
  }

  // The previous step leads to all zeros for the full tile lb and ub position
  // itself; remove those and any other duplicates / trivial redundancies.
  cst.removeTrivialRedundancy();

  // Turn everything into dims conservatively since we earlier turned all
  // trailing ids past point loop IV into symbols. Some of these could be outer
  // loop IVs; we'll canonicalize anyway.
  cst.setDimSymbolSeparation(0);

  IntegerSet ifCondSet = cst.getAsIntegerSet(context);
  // ifCondSet can be null if cst was empty -- this can happen if all loops
  // in the nest have constant trip counts.
  if (!ifCondSet)
    return nullptr;

  SmallVector<Value, 4> setOperands;
  cst.getIdValues(0, cst.getNumDimAndSymbolIds(), &setOperands);
  canonicalizeSetAndOperands(&ifCondSet, &setOperands);
  return b.create<AffineIfOp>(loops[0].getLoc(), ifCondSet, setOperands,
                              /*withElseRegion=*/true);
}

/// Create the full tile loop nest (along with its body).
static LogicalResult
createFullTiles(MutableArrayRef<AffineForOp> inputNest,
                SmallVectorImpl<AffineForOp> &fullTileLoops, OpBuilder b) {
  fullTileLoops.reserve(inputNest.size());

  // For each loop in the original nest identify a lower/upper bound pair such
  // that their difference is a constant.
  FlatAffineConstraints cst;
  for (auto loop : inputNest) {
    // TODO: straightforward to generalize to a non-unit stride.
    if (loop.getStep() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[tile separation] non-unit stride not implemented\n");
      return failure();
    }
    getIndexSet({loop}, &cst);
    // We will mark everything other than this loop IV as symbol for getting a
    // pair of <lb, ub> with a constant difference.
    cst.setDimSymbolSeparation(cst.getNumDimAndSymbolIds() - 1);
    unsigned lbPos, ubPos;
    if (!cst.getConstantBoundOnDimSize(/*pos=*/0, /*lb=*/nullptr,
                                       /*lbDivisor=*/nullptr, /*ub=*/nullptr,
                                       &lbPos, &ubPos) ||
        lbPos == ubPos) {
      LLVM_DEBUG(llvm::dbgs() << "[tile separation] Can't get constant diff / "
                                 "equalities not yet handled\n");
      return failure();
    }

    // Set all identifiers as dimensions uniformly since some of those marked as
    // symbols above could be outer loop IVs (corresponding tile space IVs).
    cst.setDimSymbolSeparation(/*newSymbolCount=*/0);

    AffineValueMap lbVmap, ubVmap;
    cst.getIneqAsAffineValueMap(/*pos=*/0, lbPos, lbVmap, b.getContext());
    cst.getIneqAsAffineValueMap(/*pos=*/0, ubPos, ubVmap, b.getContext());
    AffineForOp fullTileLoop = createCanonicalizedAffineForOp(
        b, loop.getLoc(), lbVmap.getOperands(), lbVmap.getAffineMap(),
        ubVmap.getOperands(), ubVmap.getAffineMap());
    b = fullTileLoop.getBodyBuilder();
    fullTileLoops.push_back(fullTileLoop);
  }

  // Add the body for the full tile loop nest.
  BlockAndValueMapping operandMap;
  for (auto loopEn : llvm::enumerate(inputNest))
    operandMap.map(loopEn.value().getInductionVar(),
                   fullTileLoops[loopEn.index()].getInductionVar());
  b = fullTileLoops.back().getBodyBuilder();
  for (auto &op : inputNest.back().getBody()->without_terminator())
    b.clone(op, operandMap);
  return success();
}

LogicalResult
mlir::separateFullTiles(MutableArrayRef<AffineForOp> inputNest,
                        SmallVectorImpl<AffineForOp> *fullTileNest) {
  if (inputNest.empty())
    return success();

  auto firstLoop = inputNest[0];

  // Each successive for op has to be nested in the other.
  auto prevLoop = firstLoop;
  for (auto loop : inputNest.drop_front(1)) {
    assert(loop.getParentOp() == prevLoop && "input not contiguously nested");
    prevLoop = loop;
  }

  // Create the full tile loop nest.
  SmallVector<AffineForOp, 4> fullTileLoops;
  OpBuilder b(firstLoop);
  if (failed(createFullTiles(inputNest, fullTileLoops, b))) {
    if (!fullTileLoops.empty())
      fullTileLoops.front().erase();
    return failure();
  }

  // Create and insert the version select right before the root of the nest.
  b = OpBuilder(firstLoop);
  AffineIfOp ifOp = createSeparationCondition(inputNest, b);
  if (!ifOp) {
    fullTileLoops.front().erase();
    LLVM_DEBUG(llvm::dbgs() << "All tiles are full tiles, or failure creating "
                               "separation condition\n");
    return failure();
  }

  // Move the full tile into the then block.
  Block *thenBlock = ifOp.getThenBlock();
  AffineForOp outermostFullTileLoop = fullTileLoops[0];
  thenBlock->getOperations().splice(
      std::prev(thenBlock->end()),
      outermostFullTileLoop.getOperation()->getBlock()->getOperations(),
      Block::iterator(outermostFullTileLoop));

  // Move the partial tile into the else block. The partial tile is the same as
  // the original loop nest.
  Block *elseBlock = ifOp.getElseBlock();
  elseBlock->getOperations().splice(
      std::prev(elseBlock->end()),
      firstLoop.getOperation()->getBlock()->getOperations(),
      Block::iterator(firstLoop));

  if (fullTileNest)
    *fullTileNest = std::move(fullTileLoops);

  return success();
}
