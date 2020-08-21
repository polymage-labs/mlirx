; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 2 x i64> @add_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: add_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = add <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @add_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: add_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = add <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @add_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: add_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = add <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @add_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: add_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = add <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}

define <vscale x 2 x i64> @sub_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sub_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = sub <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @sub_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sub_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = sub <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @sub_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sub_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = sub <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @sub_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sub_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = sub <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}

define <vscale x 2 x i64> @sqadd_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqadd_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqadd z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = call <vscale x 2 x i64> @llvm.sadd.sat.nxv2i64(<vscale x 2 x i64>  %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @sqadd_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqadd_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqadd z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = call <vscale x 4 x i32> @llvm.sadd.sat.nxv4i32(<vscale x 4 x i32>  %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @sqadd_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqadd_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqadd z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = call <vscale x 8 x i16> @llvm.sadd.sat.nxv8i16(<vscale x 8 x i16>  %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @sqadd_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqadd_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqadd z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = call <vscale x 16 x i8> @llvm.sadd.sat.nxv16i8(<vscale x 16 x i8>  %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}


define <vscale x 2 x i64> @sqsub_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqsub_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqsub z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = call <vscale x 2 x i64> @llvm.ssub.sat.nxv2i64(<vscale x 2 x i64>  %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @sqsub_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqsub_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqsub z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = call <vscale x 4 x i32> @llvm.ssub.sat.nxv4i32(<vscale x 4 x i32>  %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @sqsub_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqsub_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqsub z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = call <vscale x 8 x i16> @llvm.ssub.sat.nxv8i16(<vscale x 8 x i16>  %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @sqsub_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqsub_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sqsub z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = call <vscale x 16 x i8> @llvm.ssub.sat.nxv16i8(<vscale x 16 x i8>  %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}


define <vscale x 2 x i64> @uqadd_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uqadd_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqadd z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = call <vscale x 2 x i64> @llvm.uadd.sat.nxv2i64(<vscale x 2 x i64>  %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @uqadd_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uqadd_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqadd z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = call <vscale x 4 x i32> @llvm.uadd.sat.nxv4i32(<vscale x 4 x i32>  %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @uqadd_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uqadd_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqadd z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = call <vscale x 8 x i16> @llvm.uadd.sat.nxv8i16(<vscale x 8 x i16>  %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @uqadd_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uqadd_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqadd z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = call <vscale x 16 x i8> @llvm.uadd.sat.nxv16i8(<vscale x 16 x i8>  %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}


define <vscale x 2 x i64> @uqsub_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uqsub_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqsub z0.d, z0.d, z1.d
; CHECK-NEXT:    ret
  %res = call <vscale x 2 x i64> @llvm.usub.sat.nxv2i64(<vscale x 2 x i64>  %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @uqsub_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uqsub_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqsub z0.s, z0.s, z1.s
; CHECK-NEXT:    ret
  %res = call <vscale x 4 x i32> @llvm.usub.sat.nxv4i32(<vscale x 4 x i32>  %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @uqsub_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uqsub_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqsub z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %res = call <vscale x 8 x i16> @llvm.usub.sat.nxv8i16(<vscale x 8 x i16>  %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @uqsub_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uqsub_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    uqsub z0.b, z0.b, z1.b
; CHECK-NEXT:    ret
  %res = call <vscale x 16 x i8> @llvm.usub.sat.nxv16i8(<vscale x 16 x i8>  %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @mla_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: mla_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    mla z2.b, p0/m, z0.b, z1.b
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %prod = mul <vscale x 16 x i8> %a, %b
  %res = add <vscale x 16 x i8> %c, %prod
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @mla_i8_multiuse(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c, <vscale x 16 x i8>* %p) {
; CHECK-LABEL: mla_i8_multiuse:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    mul z1.b, p0/m, z1.b, z0.b
; CHECK-NEXT:    add z0.b, z2.b, z1.b
; CHECK-NEXT:    st1b { z1.b }, p0, [x0]
; CHECK-NEXT:    ret
  %prod = mul <vscale x 16 x i8> %a, %b
  store <vscale x 16 x i8> %prod, <vscale x 16 x i8>* %p
  %res = add <vscale x 16 x i8> %c, %prod
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @mls_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: mls_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    mls z2.b, p0/m, z0.b, z1.b
; CHECK-NEXT:    mov z0.d, z2.d
; CHECK-NEXT:    ret
  %prod = mul <vscale x 16 x i8> %a, %b
  %res = sub <vscale x 16 x i8> %c, %prod
  ret <vscale x 16 x i8> %res
}

declare <vscale x 16 x i8> @llvm.sadd.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.sadd.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.sadd.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.sadd.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.ssub.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.ssub.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.ssub.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.ssub.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.uadd.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.uadd.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.uadd.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.uadd.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.usub.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.usub.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.usub.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.usub.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
