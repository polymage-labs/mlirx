// RUN: mlir-opt %s -allow-unregistered-dialect -affine-scalrep | FileCheck %s

// Replacement within the innermost loop body.
// CHECK-LABEL: func @load_replace
func @load_replace() -> memref<100xf32> {
  %A = alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%i] : memref<100xf32>
    %v = affine.load %A[%i] : memref<100xf32>
    %w = affine.load %A[%i] : memref<100xf32>
    %x = addf %u, %v : f32
    %y = addf %x, %w : f32
  }
  return %A : memref<100xf32>
}
// CHECK:         affine.for %arg0 = 0 to 100 {
// CHECK-NEXT:      [[SCALAR:%[0-9]+]] = affine.load %0[%arg0] : memref<100xf32>
// CHECK-NEXT:      %{{.*}} = addf [[SCALAR]], [[SCALAR]] : f32
// CHECK-NEXT:      %{{.*}} = addf %{{.*}}, [[SCALAR]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %{{.*}} : memref<100xf32>


// Replacement + hoisting from the innermost loop.
// CHECK-LABEL: func @load_hoist
func @load_hoist() -> memref<100xf32> {
  %A = alloc() : memref<100xf32>
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      %u = affine.load %A[%i] : memref<100xf32>
      %v = affine.load %A[%i] : memref<100xf32>
      %w = affine.load %A[%i] : memref<100xf32>
      %x = addf %u, %v : f32
      %y = addf %x, %w : f32
    }
  }
  return %A : memref<100xf32>
}
// CHECK:         affine.for %arg0 = 0 to 100 {
// CHECK-NEXT:      [[SCALAR:%[0-9]+]] = affine.load %{{.*}}[%arg0] : memref<100xf32>
// CHECK-NEXT:      affine.for %arg1 = 0 to 100 {
// CHECK-NEXT:        %{{.*}} = addf [[SCALAR]], [[SCALAR]] : f32
// CHECK-NEXT:        %{{.*}} = addf %{{.*}}, [[SCALAR]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %{{.*}} : memref<100xf32>


// Replacement in the presence of loop invariant loads and stores (needs
// single element memref to hold the scalar) - typical for reductions.
// CHECK-LABEL: func @reduction
func @reduction(%B : memref<100x101xf32>) -> memref<100xf32> {
  %A = alloc() : memref<100xf32>
  %cf0 = constant 0.0 : f32

  affine.for %i = 0 to 100 {
    store %cf0, %A[%i] : memref<100xf32>
    affine.for %j = 0 to 100 {
      %a = affine.load %A[%i] : memref<100xf32>
      %b = affine.load %B[%i, %j] : memref<100x101xf32>
      %c = affine.load %B[%i, %j + 1] : memref<100x101xf32>
      %d = addf %a, %b : f32
      %e = addf %c, %d : f32
      affine.store %e, %A[%i] : memref<100xf32>
    }
  }
  return %A : memref<100xf32>
}
// CHECK:         affine.for %arg1 = 0 to 100 {
// CHECK:           store %cst, %0[%arg1] : memref<100xf32>
// CHECK-NEXT:      [[SCALBUF:%[0-9]+]] = alloca() : memref<1xf32>
// CHECK-NEXT:      %{{.*}} = affine.load %0[%arg1] : memref<100xf32>
// CHECK-NEXT:      affine.store %{{.*}}, [[SCALBUF]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg2 = 0 to 100 {
// CHECK-NEXT:        %{{.*}} = affine.load [[SCALBUF]][0] : memref<1xf32>
// CHECK-NEXT:        %{{.*}} = affine.load %arg0[%{{.*}}, %{{.*}}] : memref<100x101xf32>
// CHECK-NEXT:        %{{.*}} = affine.load %arg0[%{{.*}}, %{{.*}}] : memref<100x101xf32>
// CHECK-NEXT:        %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        affine.store %{{.*}}, [[SCALBUF]][0] : memref<1xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.*}} = affine.load [[SCALBUF]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      affine.store %{{.*}}, %0[%arg1] : memref<100xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : memref<100xf32>


// An unroll-jammed loop where multiple invariant references get separate
// registers. %A[%i], %A[%i+1] will be replaced by scalars (one element
// memref's which will subsequently be replaced by virtual registers).
// CHECK-LABEL: func @abc
func @abc(%B : memref<100x100xf32>) -> memref<100xf32> {
  %A = alloc() : memref<100xf32>
  %cf0 = constant 0.0 : f32

  affine.for %i = 0 to 100 step 2 {
    affine.store %cf0, %A[%i] : memref<100xf32>
    affine.store %cf0, %A[%i + 1] : memref<100xf32>
    affine.for %j = 0 to 100 {
      %u1 = affine.load %A[%i] : memref<100xf32>
      %u2 = affine.load %A[%i + 1] : memref<100xf32>
      %w1 = affine.load %B[%i, %j] : memref<100x100xf32>
      %w2 = affine.load %B[%i + 1, %j] : memref<100x100xf32>
      %x1 = addf %u1, %w1 : f32
      %x2 = addf %u2, %w2 : f32
      affine.store %x1, %A[%i] : memref<100xf32>
      affine.store %x2, %A[%i + 1] : memref<100xf32>
    }
  }
  return %A : memref<100xf32>
}
// CHECK:         affine.for %arg1 = 0 to 100 step 2 {
// CHECK-NEXT:      affine.store %cst, %0[%arg1] : memref<100xf32>
// CHECK-NEXT:      affine.store %cst, %0[%arg1 + 1] : memref<100xf32>
// CHECK-NEXT:      [[SCALBUF1:%[0-9]+]] = alloca() : memref<1xf32>
// CHECK-NEXT:      affine.store %cst, [[SCALBUF1]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      [[SCALBUF2:%[0-9]+]] = alloca() : memref<1xf32>
// CHECK-NEXT:      affine.store %cst, [[SCALBUF2]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg2 = 0 to 100 {
// CHECK-NEXT:        affine.load [[SCALBUF1]][0] : memref<1xf32>
// CHECK-NEXT:        affine.apply #map{{.*}}(%arg1)
// CHECK-NEXT:        affine.load [[SCALBUF2]][0] : memref<1xf32>
// CHECK-NEXT:        affine.load %arg0[%arg1, %arg2] : memref<100x100xf32>
// CHECK-NEXT:        affine.load %arg0[%arg1 + 1, %arg2] : memref<100x100xf32>
// CHECK-NEXT:        addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        affine.store %{{.*}}, [[SCALBUF1]][0] : memref<1xf32>
// CHECK-NEXT:        affine.apply #map{{.*}}(%arg1)
// CHECK-NEXT:        affine.store %{{.*}}, [[SCALBUF2]][0] : memref<1xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.load [[SCALBUF2]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      affine.store %{{.*}}, %0[%arg1 + 1] : memref<100xf32>
// CHECK-NEXT:      %{{.*}} = affine.load [[SCALBUF1]][%c0{{.*}}] : memref<1xf32>
// CHECK-NEXT:      affine.store %{{.*}}, %0[%arg1] : memref<100xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : memref<100xf32>

// Check for access equality in non-trivial cases: chain of maps / operands in
// different order.
// CHECK-LABEL: func @ref_non_trivial_equality
func @ref_non_trivial_equality(%A : memref<100 x 100 x f32>, %M : index) {
  %N = affine.apply affine_map<(d0) -> (d0 + 1)> (%M)
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      %u = affine.load %A[%i, %j] : memref<100x100xf32>
      %v = affine.load %A[%j, %i] : memref<100x100xf32>
      %idx = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %idy = affine.apply affine_map<(d0) -> (d0 - 1)>(%j)
      %w = affine.load %A[%idy + 1, %idx - 1] : memref<100x100xf32>
      %x = affine.load %A[%j mod 2 + %i - %j mod 2, %j] : memref<100x100xf32>
      %y = affine.load %A[%i, %j] : memref<100x100xf32>
      %z = affine.load %A[%i + %M, %j] : memref<100x100xf32>
      %o = affine.load %A[%i + %N - 1, %j] : memref<100x100xf32>
      "foo" (%u, %v, %w, %x, %y, %z, %o) : (f32, f32, f32, f32, f32, f32, f32) -> ()
    }
  }
  return
}
// Seven loads reduced to three here.
// CHECK:    affine.for %{{.*}} = 0 to 100 {
// CHECK-NEXT: affine.for %{{.*}} = 0 to 100 {
// CHECK-NEXT: [[SCAL1:%[0-9]+]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<100x100xf32>
// CHECK-NEXT: [[SCAL2:%[0-9]+]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<100x100xf32>
// CHECK:      [[SCAL3:%[0-9]+]] = affine.load %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}}] : memref<100x100xf32>
// CHECK-NEXT: "foo"([[SCAL1]], [[SCAL2]], [[SCAL2]], [[SCAL1]], [[SCAL1]], [[SCAL3]], [[SCAL3]])


// The loads on A[i, j] can't be replaced since there is a store to A[j, i].
// CHECK-LABEL: func @refs_not_known_to_be_equal
func @refs_not_known_to_be_equal(%A : memref<100 x 100 x f32>, %M : index) {
  %N = affine.apply affine_map<(d0) -> (d0 + 1)> (%M)
  %cf1 = constant 1.0 : f32
  // CHECK: affine.for %[[I:.*]] =
  affine.for %i = 0 to 100 {
    // CHECK-NEXT: affine.for %[[J:.*]] =
    affine.for %j = 0 to 100 {
      // CHECK: affine.load %{{.*}}[%[[I]], %[[J]]]
      %u = affine.load %A[%i, %j] : memref<100x100xf32>
      // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%[[J]], %[[I]]]
      affine.store %cf1, %A[%j, %i] : memref<100x100xf32>
      // CHECK-NEXT: affine.load %{{.*}}[%[[I]], %[[J]]]
      %v = affine.load %A[%i, %j] : memref<100x100xf32>
      // This load goes away.
      %w = affine.load %A[%j, %i] : memref<100x100xf32>
      // CHECK-NEXT: "foo"
      "foo" (%u, %v, %w) : (f32, f32, f32) -> ()
    }
  }
  return
}

// Here, no scalar replacement happens because we don't know if A[%M] and
// A[%N] are the same.
// CHECK-LABEL: func @ref_may_be_different
func @ref_may_be_different(%A : memref<100 x f32>, %M : index, %N : index) {
  %cf1 = constant 1.0 : f32
  affine.for %i = 0 to 100 {
    %u = affine.load %A[%M] : memref<100 x f32>
    %x = addf %u, %cf1 : f32
    affine.store %x, %A[%M] : memref<100 x f32>
    %v = affine.load %A[%N] : memref<100 x f32>
    %y = addf %v, %cf1 : f32
    affine.store %y, %A[%N] : memref<100 x f32>
// CHECK: affine.for {{.*}} = 0 to 100 {
// CHECK:   affine.load %arg0[%arg1] : memref<100xf32>
// CHECK:   addf %{{.*}}, %cst : f32
// CHECK:   affine.store %{{.*}}, %arg0[%arg1] : memref<100xf32>
// CHECK:   addf %{{.*}}, %cst : f32
// CHECK:   affine.store %{{.*}}, %arg0[%arg2] : memref<100xf32>
  }
  return
}

// TODO: handle this test case
// CHECK-LABEL: func @escaping_use
func @escaping_use(%A : memref<100 x f32>, %M : index, %N : index) {
  %cf1 = constant 1.0 : f32
  affine.for %i = 0 to 100 {
    // Scalar replacement should not happen here.
    affine.for %j = 0 to 100 {
      %v = affine.load %A[%i] : memref<100 x f32>
      %w = addf %v, %cf1 : f32
      affine.store %w, %A[%i] : memref<100 x f32>
      // Enable this later.
      // "foo"(%A) : (memref<100 x f32>) -> ()
    }
    // Scalar replacement shouldn't happen here.
    affine.for %j = 0 to 100 {
      %u = affine.load %A[%i] : memref<100 x f32>
      // Enable this later.
      // "foo"(%A) : (memref<100 x f32>) -> ()
      %v = affine.load %A[%i] : memref<100 x f32>
      // Enable this later.
      // "bar"(%u, %v) : (f32, f32) -> ()
    }
  }
  return
}

// The test checks for value forwarding from vector stores to vector loads.
// The value loaded from %in can directly be stored to %out by eliminating
// store and load from %tmp.
func @vector_forwarding(%in : memref<512xf32>, %out : memref<512xf32>) {
  %tmp = alloc() : memref<512xf32>
  affine.for %i = 0 to 16 {
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld0, %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    %ld1 = affine.vector_load %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}
