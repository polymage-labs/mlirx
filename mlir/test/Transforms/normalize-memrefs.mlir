// RUN: mlir-opt -allow-unregistered-dialect -normalize-memrefs %s | FileCheck %s

// CHECK-LABEL: func @matmul
// CHECK-SAME: (%[[A:arg[0-9]+]]: memref<4x4xf64>, %[[B:arg[0-9]+]]: index, %[[C:arg[0-9]+]]: memref<8xf64>) -> memref<4x4xf64>

#tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
func @matmul(%A: memref<16xf64, #tile>, %B: index, %C: memref<8xf64>) -> (memref<16xf64, #tile>) {
  affine.for %arg3 = 0 to 16 {
        %a = affine.load %A[%arg3] : memref<16xf64, #tile>
        %p = mulf %a, %a : f64
        affine.store %p, %A[%arg3] : memref<16xf64, #tile>
  }
  %c = alloc() : memref<8xf64, #tile>
  %d = affine.load %c[0] : memref<8xf64, #tile>
  return %A: memref<16xf64, #tile>
}

// CHECK-NEXT: affine.for %[[i:arg[0-9]+]]
// CHECK-NEXT: %[[a:[0-9]+]] = affine.load %[[A]][%[[i]] floordiv 4, %[[i]] mod 4] : memref<4x4xf64>
// CHECK-NEXT: %[[p:[0-9]+]] = mulf %[[a]], %[[a]] : f64
// CHECK-NEXT: affine.store %[[p]], %[[A]][%[[i]] floordiv 4, %[[i]] mod 4] : memref<4x4xf64>
// CHECK: return %[[A]] : memref<4x4xf64>