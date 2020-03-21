//===- RunnerUtils.cpp - Utils for MLIR exec on targets with a C++ runtime ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file imlpements basic functions to debug structured MLIR types at
// runtime. Entities in this file may not be compatible with targets without a
// C++ runtime. These may be progressively migrated to CRunnerUtils.cpp over
// time.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include "mlir/ExecutionEngine/RunnerUtils.h"

extern "C" void _mlir_ciface_print_memref_vector_4x4xf32(
    StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  impl::printMemRef(*M);
}

#define MEMREF_CASE(TYPE, RANK)                                                \
  case RANK:                                                                   \
    impl::printMemRef(*(static_cast<StridedMemRefType<TYPE, RANK> *>(ptr)));   \
    break

extern "C" void _mlir_ciface_print_memref_i8(UnrankedMemRefType<int8_t> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int64_t rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(int8_t, 0);
    MEMREF_CASE(int8_t, 1);
    MEMREF_CASE(int8_t, 2);
    MEMREF_CASE(int8_t, 3);
    MEMREF_CASE(int8_t, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void _mlir_ciface_print_memref_i32(UnrankedMemRefType<int32_t> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int64_t rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(int32_t, 0);
    MEMREF_CASE(int32_t, 1);
    MEMREF_CASE(int32_t, 2);
    MEMREF_CASE(int32_t, 3);
    MEMREF_CASE(int32_t, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void _mlir_ciface_print_memref_f32(UnrankedMemRefType<float> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int64_t rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(float, 0);
    MEMREF_CASE(float, 1);
    MEMREF_CASE(float, 2);
    MEMREF_CASE(float, 3);
    MEMREF_CASE(float, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void print_memref_i32(int64_t rank, void *ptr) {
  UnrankedMemRefType<int32_t> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_i32(&descriptor);
}

extern "C" void _mlir_ciface_print_memref_f64(UnrankedMemRefType<double> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(double, 0);
    MEMREF_CASE(double, 1);
    MEMREF_CASE(double, 2);
    MEMREF_CASE(double, 3);
    MEMREF_CASE(double, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void print_memref_f32(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_f32(&descriptor);
}

extern "C" void print_memref_f64(int64_t rank, void *ptr) {
  UnrankedMemRefType<double> descriptor;
  descriptor.rank = rank;
  descriptor.descriptor = ptr;
  _mlir_ciface_print_memref_f64(&descriptor);
}

extern "C" void
_mlir_ciface_print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_2d_f64(StridedMemRefType<double, 2> *M) {
  impl::printMemRef(*M);
}

extern "C" void print_flops(double flops) {
  std::cerr << flops / 1.0E9 << " GFLOPS" << std::endl;
}

extern "C" double rtclock() {
  struct timeval Tp;
  int stat = gettimeofday(&Tp, NULL);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

