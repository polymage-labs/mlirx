// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-skip-non-unit-stride-loops | FileCheck %s
// Small buffer size to trigger fine copies.
// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-fast-mem-capacity=1 | FileCheck --check-prefix=CHECK-SMALL %s

// Test affine data copy with a memref filter. We use a test pass that invokes
// affine data copy utility on the input loop nest.
// '-test-affine-data-copy-memref-filter' passes the first memref found in an
// affine.load op in the innermost loop as a filter.
// RUN: mlir-opt %s -split-input-file -test-affine-data-copy='memref-filter=1' | FileCheck %s --check-prefix=FILTER

// -copy-skip-non-stride-loops forces the copies to be placed right inside the
// tile space loops, avoiding the sensitivity of copy placement depth to memory
// footprint -- so that one could write a definite test case and not have to
// update it each time something related to the cost functions change.

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 128)>

// Map used to index the buffer while computing.
// CHECK-DAG: [[MAP_IDENTITY:map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_PLUS_128:map[0-9]+]] = affine_map<(d0) -> (d0 + 128)>

// CHECK-LABEL: func @matmul
// FILTER-LABEL: func @matmul
func @matmul(%A: memref<4096x4096xf32>, %B: memref<4096x4096xf32>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> {
  affine.for %i = 0 to 4096 step 128 {
    affine.for %j = 0 to 4096 step 128 {
      affine.for %k = 0 to 4096 step 128 {
        affine.for %ii = #map0(%i) to #map1(%i) {
          affine.for %jj = #map0(%j) to #map1(%j) {
            affine.for %kk = #map0(%k) to #map1(%k) {
              %5 = affine.load %A[%ii, %kk] : memref<4096x4096xf32>
              %6 = affine.load %B[%kk, %jj] : memref<4096x4096xf32>
              %7 = affine.load %C[%ii, %jj] : memref<4096x4096xf32>
              %8 = mulf %5, %6 : f32
              %9 = addf %7, %8 : f32
              affine.store %9, %C[%ii, %jj] : memref<4096x4096xf32>
            }
          }
        }
      }
    }
  }
  return %C : memref<4096x4096xf32>
}

// Buffers of size 128x128 get created here for all three matrices.

// CHECK: affine.for %[[I:.*]] = 0 to 4096 step 128 {
// CHECK:   affine.for %[[J:.*]] = 0 to 4096 step 128 {
// CHECK:     [[BUFC:%[0-9]+]] = alloc() : memref<128x128xf32>

// The result matrix's copy gets hoisted out.
// Result matrix copy-in.
// CHECK:     affine.for %[[II:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:       affine.for %[[JJ:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:         affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:         affine.store %{{.*}}, [[BUFC]][-%[[I]] + %[[II]], -%[[J]] + %[[JJ]]] : memref<128x128xf32>
// CHECK:       }
// CHECK:     }

// LHS matrix copy-in.
// CHECK:     affine.for %[[K:.*]] = 0 to 4096 step 128 {
// CHECK:      [[BUFA:%[0-9]+]] = alloc() : memref<128x128xf32>
// CHECK:       affine.for %[[II:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:         affine.for %[[KK:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:           affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:           affine.store %{{.*}}, [[BUFA]][-%[[I]] + %[[II]], -%[[K]] + %[[KK]]] : memref<128x128xf32>
// CHECK:         }
// CHECK:       }

// RHS matrix copy-in.
// CHECK:       [[BUFB:%[0-9]+]] = alloc() : memref<128x128xf32>
// CHECK:       affine.for %[[KK:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:         affine.for %[[JJ:.*]] = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:           affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:           affine.store %{{.*}}, [[BUFB]][-%[[K]] + %[[KK]], -%[[J]] + %[[JJ]]] : memref<128x128xf32>
// CHECK:         }
// CHECK:       }

// Computation on the fast buffers.
// CHECK:       affine.for %{{.*}} = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:         affine.for %{{.*}} = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:           affine.for %{{.*}} = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:             affine.load [[BUFA]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             affine.load [[BUFB]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             affine.load [[BUFC]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             mulf %{{.*}}, %{{.*}} : f32
// CHECK:             addf %{{.*}}, %{{.*}} : f32
// CHECK:             affine.store %{{.*}}, [[BUFC]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       dealloc [[BUFB]] : memref<128x128xf32>
// CHECK:       dealloc [[BUFA]] : memref<128x128xf32>
// CHECK:     }

// Result matrix copy out.
// CHECK:     affine.for %{{.*}} = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:       affine.for %{{.*}} = #[[MAP_IDENTITY]](%{{.*}}) to #[[MAP_PLUS_128]](%{{.*}}) {
// CHECK:         %{{.*}} = affine.load [[BUFC]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     dealloc [[BUFC]] : memref<128x128xf32>
// CHECK:   }
// CHECK: }

// Check that only one memref is copied when memref filter is used.

//      FILTER: affine.for %{{.*}} = 0 to 4096 step 128 {
//      FILTER:   alloc() : memref<128x4096xf32>
//  FILTER-NOT:   alloc()
//      FILTER:   affine.for %{{.*}} =
//      FILTER:     affine.for %{{.*}} = 0 to 4096 {
//      FILTER:   affine.for %{{.*}} = 0 to 4096 step 128 {
// FILTER-NEXT:     affine.for %{{.*}} = 0 to 4096 step 128 {
// FILTER-NEXT:       affine.for %{{.*}} = #map{{.*}}(%{{.*}}) to #map{{.*}}(%{{.*}}) {
// FILTER-NEXT:         affine.for %{{.*}} = #map{{.*}}(%{{.*}}) to #map{{.*}}(%{{.*}}) {
// FILTER-NEXT:           affine.for %{{.*}} = #map{{.*}}(%{{.*}}) to #map{{.*}}(%{{.*}}) {
//      FILTER:   dealloc %1 : memref<128x4096xf32>
//  FILTER-NOT:   dealloc %1 : memref<128x4096xf32>

// -----

//
// This test case will lead to single element buffers. These are eventually
// expected to be turned into registers via alloca and mem2reg.
//
// CHECK-SMALL-LABEL: func @foo
// FILTER-LABEL: func @foo
func @foo(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %6 = affine.load %arg1[%k, %j] : memref<1024x1024xf32>
        %7 = affine.load %arg2[%i, %j] : memref<1024x1024xf32>
        %9 = addf %6, %7 : f32
        affine.store %9, %arg2[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return %arg2 : memref<1024x1024xf32>
}
// CHECK-SMALL: affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:   affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:     %{{.*}} = alloc() : memref<1x1xf32>
// CHECK-SMALL:     %{{.*}} = affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:     affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:     affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:       %{{.*}} = alloc() : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-SMALL:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       dealloc %{{.*}} : memref<1x1xf32>
// CHECK-SMALL:     }
// CHECK-SMALL:     %{{.*}} = affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:     affine.store %{{.*}}, %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:     dealloc %{{.*}} : memref<1x1xf32>
// CHECK-SMALL:   }
// CHECK-SMALL: }
// CHECK-SMALL: return

// Check that only one memref is copied when memref filter is used.

//      FILTER: alloc() : memref<1024x1024xf32>
//  FILTER-NOT: alloc()
//      FILTER: affine.for %{{.*}} = 0 to 1024 {
//      FILTER:   affine.for %{{.*}} = 0 to 1024 {
//      FILTER: affine.for %{{.*}} = 0 to 1024 {
// FILTER-NEXT:   affine.for %{{.*}} = 0 to 1024 {
// FILTER-NEXT:     affine.for %{{.*}} = 0 to 1024 {
//      FILTER: dealloc %{{.*}} : memref<1024x1024xf32>
//  FILTER-NOT: dealloc
//  FILTER:     return

// -----

#map0 = affine_map<(d0) -> (d0)>
#map_ub = affine_map<(d0) -> (4096, d0 + 100)>

// CHECK-DAG: [[MAP_IDENTITY:map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_MIN_UB1:map[0-9]+]] = affine_map<(d0) -> (d0 + 100, 4096)>
// CHECK-DAG: [[MAP_MIN_UB2:map[0-9]+]] = affine_map<(d0) -> (4096, d0 + 100)>

// CHECK-LABEL: func @min_upper_bound
func @min_upper_bound(%A: memref<4096xf32>) -> memref<4096xf32> {
  affine.for %i = 0 to 4096 step 100 {
    affine.for %ii = #map0(%i) to min #map_ub(%i) {
      %5 = affine.load %A[%ii] : memref<4096xf32>
      %6 = mulf %5, %5 : f32
      affine.store %6, %A[%ii] : memref<4096xf32>
    }
  }
  return %A : memref<4096xf32>
}
// CHECK:    affine.for %[[IV1:.*]] = 0 to 4096 step 100
// CHECK:      %{{.*}} = alloc() : memref<100xf32>
// CHECK:      affine.for %[[IV2:.*]] = #[[MAP_IDENTITY]](%[[IV1]]) to min #[[MAP_MIN_UB1]](%[[IV1]]) {
// CHECK:        affine.load %{{.*}}[%[[IV2]]] : memref<4096xf32>
// CHECK:        affine.store %{{.*}}, %{{.*}}[-%[[IV1]] + %[[IV2]]] : memref<100xf32>
// CHECK:      }
// CHECK:      affine.for %[[IV2:.*]] = #[[MAP_IDENTITY]](%[[IV1]]) to min #[[MAP_MIN_UB2]](%[[IV1]]) {
// CHECK:        affine.load %{{.*}}[-%[[IV1]] + %[[IV2]]] : memref<100xf32>
// CHECK:        mulf %{{.*}}, %{{.*}}
// CHECK:        affine.store %{{.*}}, %{{.*}}[-%[[IV1]] + %[[IV2]]] : memref<100xf32>
// CHECK:      }
// CHECK:      affine.for %[[IV2:.*]] = #[[MAP_IDENTITY]](%[[IV1]]) to min #[[MAP_MIN_UB1]](%[[IV1]]) {
// CHECK:        affine.load %{{.*}}[-%[[IV1]] + %[[IV2]]] : memref<100xf32>
// CHECK:        affine.store %{{.*}}, %{{.*}}[%[[IV2]]] : memref<4096xf32>
// CHECK:      }
// CHECK:      dealloc %0 : memref<100xf32>
// CHECK:    }

// -----

#lb = affine_map<(d0, d1) -> (d0 * 512, d1 * 6)>
#ub = affine_map<(d0, d1) -> (d0 * 512 + 512, d1 * 6 + 6)>

// A buffer of size 2048 x 6 should be created here; it should be
// indexed by jj - 6 * j. This pattern typically appears with
// multi-level tiling when the tile sizes used don't divide the memref
// extent.

// CHECK-LABEL: lower_bound_max(%{{.*}}: memref<2048x516xf64>,
// CHECK-SAME: [[i:arg[0-9]+]]
// CHECK-SAME: [[j:arg[0-9]+]]
func @lower_bound_max(%M: memref<2048x516xf64>, %i : index, %j : index) {
  affine.for %ii = 0 to 2048 {
    affine.for %jj = max #lb(%i, %j) to min #ub(%i, %j) {
      affine.load %M[%ii, %jj] : memref<2048x516xf64>
    }
  }
  return
}

// CHECK: alloc() : memref<2048x6xf64>
// CHECK: affine.for %[[ii:.*]] = 0 to 2048 {
// CHECK:   affine.for %[[jj:.*]] = max #map{{.*}}()[%[[i]], %[[j]]] to min #map{{.*}}()[%[[i]], %[[j]]] {
// CHECK:      affine.load %{{.*}}[%[[ii]], %[[jj]]] : memref<2048x516xf64>
// CHECK:      affine.store %{{.*}}, %0[%[[ii]], %[[jj]] - symbol(%[[j]]) * 6] : memref<2048x6xf64>
// CHECK:   }
// CHECK: }
// CHECK: affine.for %{{.*}} = 0 to 2048 {
// CHECK:   affine.for %{{.*}} = max #map{{.*}}()[%{{.*}}, %{{.*}}] to min #map{{.*}}()[%{{.*}}, %{{.*}}] {
// CHECK:     affine.load %0[%{{.*}}, %{{.*}} - symbol(%{{.*}}) * 6] : memref<2048x6xf64>
// CHECK:    }
// CHECK: }
// CHECK: dealloc %{{.*}} : memref<2048x6xf64>
