// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: @arbitrary_bound
func @arbitrary_bound(%n : index) {
  affine.for %i = 0 to %n {
    affine.graybox [] = () : () -> () {
      // %pow can now be used as a loop bound.
      %pow = call @powi(%i) : (index) ->  index
      affine.for %j = 0 to %pow {
        "foo"() : () -> ()
      }
      return
    }
    // CHECK:      affine.graybox [] = () : () -> () {
    // CHECK-NEXT:   call @powi
    // CHECK-NEXT:   affine.for
    // CHECK-NEXT:     "foo"()
    // CHECK-NEXT:   }
    // CHECK-NEXT:   return
    // CHECK-NEXT: }
  }
  return
}

func @powi(index) -> index

// CHECK-LABEL: func @arbitrary_mem_access
func @arbitrary_mem_access(%I: memref<128xi32>, %M: memref<1024xf32>) {
  affine.for %i = 0 to 128 {
    // CHECK: %{{.*}} = affine.graybox [{{.*}}] = ({{.*}}) : (memref<128xi32>, memref<1024xf32>) -> f32
    affine.graybox [%rI, %rM] = (%I, %M) : (memref<128xi32>, memref<1024xf32>) -> f32 {
      %idx = affine.load %rI[%i] : memref<128xi32>
      %index = index_cast %idx : i32 to index
      %v = affine.load %rM[%index]: memref<1024xf32>
      return %v : f32
    }
  }
  return
}

// CHECK-LABEL: @symbol_check
func @symbol_check(%B: memref<100xi32>, %A: memref<100xf32>) {
  %cf1 = constant 1.0 : f32
  affine.for %i = 0 to 100 {
    %v = affine.load %B[%i] : memref<100xi32>
    %vo = index_cast %v : i32 to index
    // CHECK: affine.graybox [%{{.*}}] = (%{{.*}}) : (memref<100xf32>) -> () {
    affine.graybox [%rA] = (%A) : (memref<100xf32>) -> () {
      // %vi is now a symbol here.
      %vi = index_cast %v : i32 to index
      affine.load %rA[%vi] : memref<100xf32>
      // %vo is also a symbol (dominates the graybox).
      affine.load %rA[%vo] : memref<100xf32>
      return
    }
    // CHECK:        index_cast
    // CHECK-NEXT:   affine.load
    // CHECK-NEXT:   affine.load
    // CHECK-NEXT:   return
    // CHECK-NEXT: }
  }
  return
}

// CHECK-LABEL: func @search
func @search(%A : memref<?x?xi32>, %S : memref<?xi32>, %key : i32) {
  %ni = dim %A, 0 : memref<?x?xi32>
  %c1 = constant 1 : index
  // This loop can be parallelized.
  affine.for %i = 0 to %ni {
    // CHECK: affine.graybox
    affine.graybox [%rA, %rS] = (%A, %S) : (memref<?x?xi32>, memref<?xi32>) -> () {
      %c0 = constant 0 : index
      %nj = dim %rA, 1 : memref<?x?xi32>
      br ^bb1(%c0 : index)

    ^bb1(%j: index):
      %p1 = cmpi "slt", %j, %nj : index
      cond_br %p1, ^bb2(%j : index), ^bb5

    ^bb2(%j_arg : index):
      %v = affine.load %rA[%i, %j_arg] : memref<?x?xi32>
      %p2 = cmpi "eq", %v, %key : i32
    cond_br %p2, ^bb3(%j_arg : index), ^bb4(%j_arg : index)

    ^bb3(%j_arg2: index):
      %j_int = index_cast %j_arg2 : index to i32
      affine.store %j_int, %rS[%i] : memref<?xi32>
      br ^bb5

    ^bb4(%j_arg3 : index):
      %jinc = addi %j_arg3, %c1 : index
      br ^bb1(%jinc : index)

    ^bb5:
      return
    }
  }
  return
}
