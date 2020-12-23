// RUN: mlir-opt -hopt -convert-linalg-to-loops -lower-affine -convert-std-to-llvm %s | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func @main() {
  %A = alloc() : memref<2088x2048xf64>
  // Align %B and %C since these are shape cast to vector types.
  %B = alloc() {alignment = 32} : memref<2048x2048xf64>
  %C = alloc() {alignment = 32} : memref<2088x2048xf64>

  %cf1 = constant 1.00000e+00 : f64

  linalg.fill(%A, %cf1) : memref<2088x2048xf64>, f64
  linalg.fill(%B, %cf1) : memref<2048x2048xf64>, f64

  %reps = constant 5 : index

  %t_start = call @rtclock() : () -> (f64)
  affine.for %ti = 0 to %reps {
    linalg.fill(%C, %cf1) : memref<2088x2048xf64>, f64
    call @matmul_hop(%A, %B, %C) : (memref<2088x2048xf64>, memref<2048x2048xf64>, memref<2088x2048xf64>) -> ()
  }
  %t_end = call @rtclock() : () -> (f64)
  call @print_memref_2d_f64(%C): (memref<2088x2048xf64>) -> ()

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = dim %C, %c0 : memref<2088x2048xf64>
  %N = dim %C, %c1 : memref<2088x2048xf64>
  %K = dim %A, %c1 : memref<2088x2048xf64>

  %t = subf %t_end, %t_start : f64
  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index
  // 2*M*N*K.
  %c2 = constant 2 : index
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}
// CHECK: 2049,   2049,   2049,

func private @print_flops(f64) -> ()
func private @rtclock() -> (f64)
func private @print_memref_2d_f64(memref<2088x2048xf64>)

func @matmul_hop(%arg0: memref<2088x2048xf64>, %arg1: memref<2048x2048xf64>,
          %arg2: memref<2088x2048xf64>) {
  "hop.matmul"(%arg0, %arg1, %arg2) {
    M_C = 330 : i32, N_C = 2048 : i32, K_C = 480 : i32, M_R = 3, N_R = 16 : i32, K_U = 4 : i32
  } : (memref<2088x2048xf64>, memref<2048x2048xf64>, memref<2088x2048xf64>) -> ()
  return
}
