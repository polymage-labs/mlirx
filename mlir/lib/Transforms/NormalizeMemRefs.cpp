//===- NormalizeMemRefs.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an interprocedural pass to normalize memref layouts.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/Passes.h"
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

#define DEBUG_TYPE "normalize-memrefs"

using namespace mlir;

/// All interprocedural memrefs with non-trivial layout maps are converted to
/// ones with trivial identity layout ones.

// Input :-
// #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
// func @matmul(%A: memref<16xf64, #tile>, %B: index, %C: memref<16xf64>) ->
// (memref<16xf64, #tile>) {
//   affine.for %arg3 = 0 to 16 {
//         %a = affine.load %A[%arg3] : memref<16xf64, #tile>
//         %p = mulf %a, %a : f64
//         affine.store %p, %A[%arg3] : memref<16xf64, #tile>
//   }
//   %c = alloc() : memref<16xf64, #tile>
//   %d = affine.load %c[0] : memref<16xf64, #tile>
//   return %A: memref<16xf64, #tile>
// }

// Output :-
// module {
//   func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
//   -> memref<4x4xf64> {
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

namespace {

struct NormalizeMemRefs : public NormalizeMemRefsBase<NormalizeMemRefs> {
  void runOnOperation() override;
  void runOnFunction(FuncOp);
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createNormalizeMemRefsPass() {
  return std::make_unique<NormalizeMemRefs>();
}

void NormalizeMemRefs::runOnOperation() {
  // Here we get hold of the module/operation that basically contains one region.
  ModuleOp moduleOp = getOperation();

  // We traverse each functions within the module in order to normalize the memref
  // type arguments.
  moduleOp.walk([&](FuncOp funcOp) {
    runOnFunction(funcOp);
  });
}

void NormalizeMemRefs::runOnFunction(FuncOp funcOp) {
  // Turn memrefs' non-identity layouts maps into ones with identity. Collect
  // alloc ops first and then process since normalizeMemRef replaces/erases ops
  // during memref rewriting.
  SmallVector<AllocOp, 4> allocOps;
  funcOp.walk([&](AllocOp op) { allocOps.push_back(op); });
  for (auto allocOp : allocOps) {
    normalizeMemRef(allocOp);
  }
  // We use this OpBuilder to create new memref layout later.
  OpBuilder b(funcOp);

  // Getting hold of the signature of the function before normalizing the inputs.
  FunctionType ft = funcOp.getType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;

  // Populating results with function's initial result (type) as
  // this will be used to modify and set function's signature later.
  for (unsigned retIndex : llvm::seq<unsigned>(0, funcOp.getNumResults())) {
    resultTypes.push_back(ft.getResult(retIndex));
  }

  // Walk over each argument of a function to perform memref normalization (if
  // any).
  for (unsigned argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    Type argType = funcOp.getArgument(argIndex).getType();
    MemRefType memrefType = argType.dyn_cast<MemRefType>();
    // Check whether argument is of MemRef type.
    if (!memrefType) {
      // Any other argument type can simply be part of the final function
      // signature.
      argTypes.push_back(argType);
      continue;
    }
    unsigned rank = memrefType.getRank();
    if (rank == 0)
      continue;

    ArrayRef<AffineMap> layoutMaps = memrefType.getAffineMaps();
    if (layoutMaps.empty() || layoutMaps.front() == b.getMultiDimIdentityMap(rank)) {
      // Either no maps is associated with this memref or this memref has
      // a trivial (identity) map. Simply add this as the function argument.
      argTypes.push_back(memrefType);
      continue;
    }

    // We don't do any checks for one-to-one'ness; we assume that it is
    // one-to-one.

    // TODO: Only for static memref's for now.
    if (memrefType.getNumDynamicDims() > 0)
      continue;

    // We have a single map that is not an identity map. Create a new memref
    // with the right shape and an identity layout map.
    ArrayRef<int64_t> shape = memrefType.getShape();
    // FlatAffineConstraint may later on use symbolicOperands.
    FlatAffineConstraints fac(rank, 0);
    for (unsigned d = 0; d < rank; ++d) {
      fac.addConstantLowerBound(d, 0);
      fac.addConstantUpperBound(d, shape[d] - 1);
    }

    // We compose this map with the original index (logical) space to derive
    // the upper bounds for the new index space.
    AffineMap layoutMap = layoutMaps.front();
    unsigned newRank = layoutMap.getNumResults();
    if (failed(fac.composeMatchingMap(layoutMap)))
      continue;
    // TODO: Handle semi-affine maps.

    // Project out the old data dimensions.
    fac.projectOut(newRank,
                   fac.getNumIds() - newRank - fac.getNumLocalIds());
    SmallVector<int64_t, 4> newShape(newRank);
    for (unsigned d = 0; d < newRank; ++d) {
      // The lower bound for the shape is always zero.
      auto ubConst = fac.getConstantUpperBound(d);
      // For a static memref and an affine map with no symbols, this is
      // always bounded.
      assert(ubConst.hasValue() && "should always have an upper bound");
      if (ubConst.getValue() < 0)
        // This is due to an invalid map that maps to a negative space.
        continue;
      newShape[d] = ubConst.getValue() + 1;
    }

    // Create the new memref type after trivializing the old layout map.
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType)
            .setShape(newShape)
            .setAffineMaps(b.getMultiDimIdentityMap(newRank));

    

    // Insert a new temporary argument with the new memref type.
    funcOp.front().insertArgument(argIndex, newMemRefType);
    BlockArgument newMemRef = funcOp.getArgument(argIndex);
    BlockArgument oldMemRef = funcOp.getArgument(argIndex+1);
    // Replace all uses of the old memref.
    if (failed(replaceAllMemRefUsesWith(oldMemRef, /*newMemRef=*/newMemRef,
                                        /*extraIndices=*/{},
                                        /*indexRemap=*/layoutMap,
                                        /*extraOperands=*/{},
                                        /*symbolOperands=*/{},
                                        /*domInstFilter=*/nullptr,
                                        /*postDomInstFilter=*/nullptr,
                                        /*allowNonDereferencingOps=*/true))) {
      // If it failed (due to escapes for example), bail out. Removing the
      // temporary argument inserted previously.
      funcOp.eraseArgument(argIndex);
      continue;
    }

    // Since in this pass the objective is to normalize the layout maps of 
    // the memref arguments and replace the uses accordingly, we 
    // check if the function return type uses the same old memref type.
    // TODO(avarmapml): Check - A function's return type might have a
    // different memref layout and a map.
    for (unsigned retIndex : llvm::seq<unsigned>(0, funcOp.getNumResults())) {
      if (resultTypes[retIndex] == memrefType)
        resultTypes[retIndex] = newMemRef.getType();
    }

    // All uses for the argument with old memref type were replaced
    // successfully. So we remove the old argument now.
    // TODO(avarmapml): replaceAllUsesWith.
    funcOp.eraseArgument(argIndex + 1);

    // Add the new type to the function signature later.
    argTypes.push_back(static_cast<Type>(newMemRefType));
  }

  ArrayRef<Type> inputs = makeArrayRef(argTypes);
  ArrayRef<Type> results = makeArrayRef(resultTypes);
  // We create a new function type and modify the function signature with this
  // new type.
  FunctionType newFuncType = FunctionType::get(/*inputs=*/inputs,
                                         /*results=*/results,
                                         /**context=*/&getContext());
  funcOp.setType(newFuncType);
}
