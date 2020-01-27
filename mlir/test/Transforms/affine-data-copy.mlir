// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-skip-non-unit-stride-loops | FileCheck %s
// Small buffer size to trigger fine copies.
// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-fast-mem-capacity=1 | FileCheck --check-prefix=CHECK-SMALL %s

// -copy-skip-non-stride-loops forces the copies to be placed right inside the
// tile space loops, avoiding the sensitivity of copy placement depth to memory
// footprint -- so that one could write a definite test case and not have to
// update it each time something related to the cost functions change.

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 128)>

// Map used to index the buffer while computing.
// CHECK-DAG: [[BUF_IDX_MAP:map[0-9]+]] = affine_map<(d0, d1, d2, d3) -> (-d0 + d1, -d2 + d3)>

// CHECK-DAG: [[MAP_IDENTITY:map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_PLUS_128:map[0-9]+]] = affine_map<(d0) -> (d0 + 128)>

// CHECK-LABEL: func @matmul
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

//
// This test case will lead to single element buffers. These are eventually
// expected to be turned into registers via alloca and mem2reg.
//
// CHECK-SMALL: func @small_mem
// CHECK-LABEL: func @small_mem
func @small_mem(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
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
