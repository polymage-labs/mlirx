//===- MyNormalizeMemRef.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to simplify affine structures in operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG_TYPE "my-normalize-memref"

using namespace mlir;

namespace {

/// Simplifies affine maps and sets appearing in the operations of the Function.
/// This part is mainly to test the simplifyAffineExpr method. In addition,
/// all memrefs with non-trivial layout maps are converted to ones with trivial
/// identity layout ones.
struct MyNormalizeMemRef
    : public MyNormalizeMemRefBase<MyNormalizeMemRef> {
  void runOnOperation() override;
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createMyNormalizeMemRefPass() {
  return std::make_unique<MyNormalizeMemRef>();
}

void MyNormalizeMemRef::runOnOperation() {
  auto func = getOperation();

  llvm::dbgs()<<" ** Before entering the walk ** \n";
  func.walk([&](FuncOp funcOp){
    SmallVector<Type, 8> inputs;
    SmallVector<Type, 4> results;
    llvm::dbgs()<<funcOp.getType()<<" *-*-\n";
    OpBuilder b(funcOp);
    llvm::dbgs()<<(int)funcOp.getNumArguments()<<":\n";
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {

      if(funcOp.getArgument(argIndex).getType().getKind() == StandardTypes::MemRef) {
        llvm::dbgs()<<argIndex<<" is a MemRef\n";
        llvm::dbgs()<<"argument type = "<<funcOp.getArgument(argIndex).getType();
        // funcOp.getArgument(argIndex).setType(UnrankedTensorType);
        MemRefType memrefType = funcOp.getArgument(argIndex).getType().cast<MemRefType>();
        unsigned rank = memrefType.getRank();
        llvm::dbgs()<<", Rank = "<<rank<<"\n";
        // if (rank == 0)
        //   return success();
        auto layoutMaps = memrefType.getAffineMaps();
        // OpBuilder b(allocOp);
        if (layoutMaps.size() != 1)
          // return failure();
          continue;

        AffineMap layoutMap = layoutMaps.front();
        llvm::dbgs()<<", AffineMap = "<<layoutMap;
        llvm::dbgs()<<", b.getMultiDimIdentityMap(rank) = "<<b.getMultiDimIdentityMap(rank);
        
        if (layoutMap == b.getMultiDimIdentityMap(rank))
          // return success();
          continue;

        if (memrefType.getNumDynamicDims() > 0)
          // return failure();
          continue;

        auto shape = memrefType.getShape();
        // FlatAffineConstraints fac(rank, allocOp.getNumSymbolicOperands());
        FlatAffineConstraints fac(rank, 0);
        for (unsigned d = 0; d < rank; ++d) {
          fac.addConstantLowerBound(d, 0);
          fac.addConstantUpperBound(d, shape[d] - 1);
        }

        unsigned newRank = layoutMap.getNumResults();
        if (failed(fac.composeMatchingMap(layoutMap)))
          continue;
          // TODO: semi-affine maps.
          // return failure();

        fac.projectOut(newRank, fac.getNumIds() - newRank - fac.getNumLocalIds());
        SmallVector<int64_t, 4> newShape(newRank);
        for (unsigned d = 0; d < newRank; ++d) {
          // The lower bound for the shape is always zero.
          auto ubConst = fac.getConstantUpperBound(d);
          // For a static memref and an affine map with no symbols, this is always
          // bounded.
          assert(ubConst.hasValue() && "should always have an upper bound");
          if (ubConst.getValue() < 0)
            // This is due to an invalid map that maps to a negative space.
            // return failure();
            continue;
          newShape[d] = ubConst.getValue() + 1;
        }

        // auto oldMemRef = allocOp.getResult();
        // SmallVector<Value, 4> symbolOperands(allocOp.getSymbolicOperands());
        MemRefType newMemRefType =
        MemRefType::Builder(memrefType)
          .setShape(newShape)
          .setAffineMaps(b.getMultiDimIdentityMap(newRank));
        // funcOp.getArgument(argIndex).setType(newMemRefType);
        llvm::dbgs()<<"Output should be of type: "<<static_cast<Type>(newMemRefType)<<"\n";
        funcOp.getArgument(argIndex).setType(static_cast<Type>(newMemRefType));
      }
    }
    // function.dump();
  });
}
