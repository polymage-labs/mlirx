/// Checks for the buffer alignment attribute in the alloc instruction.
// RUN: mlir-opt %s -test-affine-data-copy="alloc-alignment" -split-input-file | FileCheck %s
// CHECK-LABEL: func @simple_gemm
// CHECK: %{{[0-9]+}} = alloc() {alignment = {{[0-9]+}} : i64} : memref<{{[0-9]+}}x{{[0-9]+}}xf32>
func @simple_gemm(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
 return
}

