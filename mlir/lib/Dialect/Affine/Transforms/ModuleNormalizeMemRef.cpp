//===- ModuleNormalizeMemRef.cpp ---------------------------------------===//
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


#define DEBUG_TYPE "module-normalize-memref"

using namespace mlir;

namespace {

/// All memrefs with non-trivial layout maps are converted to ones with trivial
/// identity layout ones - interprocedural

// Input :-
// #pad = affine_map<(i) -> (i floordiv 4, i mod 4)>
// func @matmul(%A: memref<16xf64, #pad>, %B: index, %C: memref<16xf64>) -> (memref<16xf64, #pad>) {
//   affine.for %arg3 = 0 to 16 {
//         %a = affine.load %A[%arg3] : memref<16xf64, #pad>
//         %p = mulf %a, %a : f64
//         affine.store %p, %A[%arg3] : memref<16xf64, #pad>
//   }
//   %c = alloc() : memref<16xf64, #pad>
//   %d = affine.load %c[0] : memref<16xf64, #pad>
//   return %A: memref<16xf64, #pad>
// }

// Output :-
// module {
//   func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>) -> memref<4x4xf64> {
//     affine.for %arg3 = 0 to 16 {
//       %2 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4] : memref<4x4xf64>
//       %3 = mulf %2, %2 : f64
//       affine.store %3, %arg0[%arg3 floordiv 4, %arg3 mod 4] : memref<4x4xf64>
//     }
//     %0 = alloc() : memref<16xf64, #map0>
//     %1 = affine.load %0[0] : memref<16xf64, #map0>
//     return %arg0 : memref<4x4xf64>
//   }
// }

struct ModuleNormalizeMemRef
    : public ModuleNormalizeMemRefBase<ModuleNormalizeMemRef> {
  void runOnOperation() override;
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createModuleNormalizeMemRefPass() {
  return std::make_unique<ModuleNormalizeMemRef>();
}

void ModuleNormalizeMemRef::runOnOperation() {
  auto func = getOperation();

  func.walk([&](FuncOp funcOp){
    OpBuilder b(funcOp);

    FunctionType ft = funcOp.getType();
    SmallVector<Type, 8> inputs; // Holds function arguments
    SmallVector<Type, 4> results; // Holds function results

    // Populating results with function's initial result (type)
    // Will be used to modify and set function's signature
    for (auto retIndex : llvm::seq<unsigned>(0, funcOp.getNumResults())) {
      results.push_back(ft.getResult(retIndex));
    }

    // Walk over each argument of a function to perform memref normalization (if any)
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {

      // Check whether argument is of MemRef type
      // TODO(avarmapml): What about unranked memrefs? 
      if(funcOp.getArgument(argIndex).getType().getKind() == StandardTypes::MemRef) {

        MemRefType memrefType = funcOp.getArgument(argIndex).getType().cast<MemRefType>();
        unsigned rank = memrefType.getRank();
        if (rank == 0)
          continue;

        auto layoutMaps = memrefType.getAffineMaps();
        if (layoutMaps.size() != 1) {
          // No maps associated with this memref, simply add this as the function argument
          inputs.push_back(funcOp.getArgument(argIndex).getType());
          continue;
        }

        AffineMap layoutMap = layoutMaps.front();
        
        if (layoutMap == b.getMultiDimIdentityMap(rank)) {
          // Memref has identity map. Already trivial, simply add this as the function argument
          inputs.push_back(funcOp.getArgument(argIndex).getType());
          continue;
        }

        // We don't do any checks for one-to-one'ness; we assume that it is
        // one-to-one.

        // TODO: Only for static memref's for now.
        if (memrefType.getNumDynamicDims() > 0)
          continue;

        // We have a single map that is not an identity map. Create a new memref with
        // the right shape and an identity layout map.
        auto shape = memrefType.getShape();
        // FlatAffineConstraints fac(rank, allocOp.getNumSymbolicOperands());
        FlatAffineConstraints fac(rank, 0);
        for (unsigned d = 0; d < rank; ++d) {
          fac.addConstantLowerBound(d, 0);
          fac.addConstantUpperBound(d, shape[d] - 1);
        }

        // We compose this map with the original index (logical) space to derive the
        // upper bounds for the new index space.
        unsigned newRank = layoutMap.getNumResults();
        if (failed(fac.composeMatchingMap(layoutMap)))
          continue;
          // TODO: semi-affine maps.

        // Project out the old data dimensions.
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

        // Create the new memref type
        MemRefType newMemRefType =
        MemRefType::Builder(memrefType)
          .setShape(newShape)
          .setAffineMaps(b.getMultiDimIdentityMap(newRank));

        auto oldMemRef = new Value(funcOp.getArgument(argIndex));

        // Insert a new temporary argument with the new memref type
        funcOp.front().insertArgument(argIndex,static_cast<Type>(newMemRefType));
        auto newMemRef = funcOp.getArgument(argIndex);

        // Replace all uses of the old memref
        if (failed(replaceAllMemRefUsesWith(*oldMemRef, /*newMemRef=*/newMemRef,
                                            /*extraIndices=*/{},
                                            /*indexRemap=*/layoutMap,
                                            /*extraOperands=*/{},
                                            /*symbolOperands=*/{}))) {
          // If it failed (due to escapes for example), bail out.
          // Removing the temporary argument inserted previously
          funcOp.front().eraseArgument(argIndex);
          continue;
        }

        // Since in this pass the objective is to normalize arguments of a function
        // and replace the uses accordingly, we check if the function return type
        // uses the same old memref type
        // TODO(avarmapml): Check - A function's return type might have a different memref layout
        // and a map.
        for (auto retIndex : llvm::seq<unsigned>(0, funcOp.getNumResults())) {
          if(results[retIndex] == (oldMemRef->getType())) {
            results[retIndex] = newMemRef.getType();
          }
        }

        // All uses for the argument with old memref type were replaced successfully
        // So remove the old argument
        // TODO(avarmapml): replaceAllUsesWith
        funcOp.front().eraseArgument(argIndex+1);

        // Add the new type to the function signature later
        inputs.push_back(static_cast<Type>(newMemRefType));
        // break;
      } else {
        // Any other argument type can simply be part of the final function signature
        inputs.push_back(funcOp.getArgument(argIndex).getType());
      }
    }
    
    ArrayRef<Type> inp = makeArrayRef(inputs);
    ArrayRef<Type> res = makeArrayRef(results);
    // Creating new function type and modifying the function signature
    FunctionType newFT = FunctionType::get(inp, res, &getContext());
    funcOp.setType(newFT);
  });
}
