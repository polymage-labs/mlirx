//===- AffineParallelize.cpp - Affineparallelize Pass -------------------------===//
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
// This file implements a parallelizer for affine loop nests that is able to
// perform inner or outer loop parallelization.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-parallel"

using namespace mlir;

namespace {
// Convert all parallel affine forOp into 1-D affine parallelOp.
struct AffineParallelize : public AffineParallelizeBase<AffineParallelize> {
#include "mlir/Dialect/Affine/Passes.h.inc"
  void runOnFunction() override;
};
} // end anonymous namespace

void AffineParallelize::runOnFunction() {
  FuncOp f = getFunction();
  OpBuilder b(f.getBody());
  SmallVector<Operation *, 8> parallelizableLoops;
  f.walk([&](AffineForOp loop) {
    if (isLoopParallel(loop)) {
      parallelizableLoops.push_back(loop);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "\n******************************************");
  LLVM_DEBUG(llvm::dbgs() << "\nParallelizing loop\n");
  for (auto loop : parallelizableLoops) {
    affineParallelize(cast<AffineForOp>(loop));
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineParallelizePass() {
  return std::make_unique<AffineParallelize>();
}

