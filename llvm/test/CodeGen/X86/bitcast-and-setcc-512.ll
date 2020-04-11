; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.2 | FileCheck %s --check-prefixes=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefixes=AVX12,AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefixes=AVX12,AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f | FileCheck %s --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=AVX512 --check-prefix=AVX512BW

define i8 @v8i64(<8 x i64> %a, <8 x i64> %b, <8 x i64> %c, <8 x i64> %d) {
; SSE-LABEL: v8i64:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtq %xmm7, %xmm3
; SSE-NEXT:    pcmpgtq %xmm6, %xmm2
; SSE-NEXT:    packssdw %xmm3, %xmm2
; SSE-NEXT:    pcmpgtq %xmm5, %xmm1
; SSE-NEXT:    pcmpgtq %xmm4, %xmm0
; SSE-NEXT:    packssdw %xmm1, %xmm0
; SSE-NEXT:    packssdw %xmm2, %xmm0
; SSE-NEXT:    pcmpgtq {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtq {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    packssdw %xmm11, %xmm10
; SSE-NEXT:    pcmpgtq {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    pcmpgtq {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    packssdw %xmm9, %xmm8
; SSE-NEXT:    packssdw %xmm10, %xmm8
; SSE-NEXT:    pand %xmm0, %xmm8
; SSE-NEXT:    packsswb %xmm0, %xmm8
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    # kill: def $al killed $al killed $eax
; SSE-NEXT:    retq
;
; AVX1-LABEL: v8i64:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vextractf128 $1, %ymm7, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm5, %xmm9
; AVX1-NEXT:    vpcmpgtq %xmm8, %xmm9, %xmm8
; AVX1-NEXT:    vpcmpgtq %xmm7, %xmm5, %xmm5
; AVX1-NEXT:    vpackssdw %xmm8, %xmm5, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm6, %xmm7
; AVX1-NEXT:    vextractf128 $1, %ymm4, %xmm5
; AVX1-NEXT:    vpcmpgtq %xmm7, %xmm5, %xmm5
; AVX1-NEXT:    vpcmpgtq %xmm6, %xmm4, %xmm4
; AVX1-NEXT:    vpackssdw %xmm5, %xmm4, %xmm4
; AVX1-NEXT:    vinsertf128 $1, %xmm8, %ymm4, %ymm4
; AVX1-NEXT:    vextractf128 $1, %ymm3, %xmm5
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm6
; AVX1-NEXT:    vpcmpgtq %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpcmpgtq %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpackssdw %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm5
; AVX1-NEXT:    vpcmpgtq %xmm3, %xmm5, %xmm3
; AVX1-NEXT:    vpcmpgtq %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpackssdw %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX1-NEXT:    vmovmskps %ymm0, %eax
; AVX1-NEXT:    # kill: def $al killed $al killed $eax
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: v8i64:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpcmpgtq %ymm7, %ymm5, %ymm5
; AVX2-NEXT:    vpcmpgtq %ymm6, %ymm4, %ymm4
; AVX2-NEXT:    vpackssdw %ymm5, %ymm4, %ymm4
; AVX2-NEXT:    vpcmpgtq %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vpcmpgtq %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpackssdw %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,1,3]
; AVX2-NEXT:    vmovmskps %ymm0, %eax
; AVX2-NEXT:    # kill: def $al killed $al killed $eax
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: v8i64:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vpcmpgtq %zmm1, %zmm0, %k1
; AVX512F-NEXT:    vpcmpgtq %zmm3, %zmm2, %k0 {%k1}
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    # kill: def $al killed $al killed $eax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v8i64:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vpcmpgtq %zmm1, %zmm0, %k1
; AVX512BW-NEXT:    vpcmpgtq %zmm3, %zmm2, %k0 {%k1}
; AVX512BW-NEXT:    kmovd %k0, %eax
; AVX512BW-NEXT:    # kill: def $al killed $al killed $eax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = icmp sgt <8 x i64> %a, %b
  %x1 = icmp sgt <8 x i64> %c, %d
  %y = and <8 x i1> %x0, %x1
  %res = bitcast <8 x i1> %y to i8
  ret i8 %res
}

define i8 @v8f64(<8 x double> %a, <8 x double> %b, <8 x double> %c, <8 x double> %d) {
; SSE-LABEL: v8f64:
; SSE:       # %bb.0:
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    cmpltpd %xmm3, %xmm7
; SSE-NEXT:    cmpltpd %xmm2, %xmm6
; SSE-NEXT:    packssdw %xmm7, %xmm6
; SSE-NEXT:    cmpltpd %xmm1, %xmm5
; SSE-NEXT:    cmpltpd %xmm0, %xmm4
; SSE-NEXT:    packssdw %xmm5, %xmm4
; SSE-NEXT:    packssdw %xmm6, %xmm4
; SSE-NEXT:    cmpltpd {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    cmpltpd {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    packssdw %xmm11, %xmm10
; SSE-NEXT:    cmpltpd {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    cmpltpd {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    packssdw %xmm9, %xmm8
; SSE-NEXT:    packssdw %xmm10, %xmm8
; SSE-NEXT:    pand %xmm4, %xmm8
; SSE-NEXT:    packsswb %xmm0, %xmm8
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    # kill: def $al killed $al killed $eax
; SSE-NEXT:    retq
;
; AVX1-LABEL: v8f64:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vcmpltpd %ymm5, %ymm7, %ymm5
; AVX1-NEXT:    vextractf128 $1, %ymm5, %xmm7
; AVX1-NEXT:    vpackssdw %xmm7, %xmm5, %xmm5
; AVX1-NEXT:    vcmpltpd %ymm4, %ymm6, %ymm4
; AVX1-NEXT:    vextractf128 $1, %ymm4, %xmm6
; AVX1-NEXT:    vpackssdw %xmm6, %xmm4, %xmm4
; AVX1-NEXT:    vinsertf128 $1, %xmm5, %ymm4, %ymm4
; AVX1-NEXT:    vcmpltpd %ymm1, %ymm3, %ymm1
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpackssdw %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vcmpltpd %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpackssdw %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX1-NEXT:    vmovmskps %ymm0, %eax
; AVX1-NEXT:    # kill: def $al killed $al killed $eax
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: v8f64:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vcmpltpd %ymm5, %ymm7, %ymm5
; AVX2-NEXT:    vcmpltpd %ymm4, %ymm6, %ymm4
; AVX2-NEXT:    vpackssdw %ymm5, %ymm4, %ymm4
; AVX2-NEXT:    vcmpltpd %ymm1, %ymm3, %ymm1
; AVX2-NEXT:    vcmpltpd %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vpackssdw %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,1,3]
; AVX2-NEXT:    vmovmskps %ymm0, %eax
; AVX2-NEXT:    # kill: def $al killed $al killed $eax
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: v8f64:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vcmpltpd %zmm0, %zmm1, %k1
; AVX512F-NEXT:    vcmpltpd %zmm2, %zmm3, %k0 {%k1}
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    # kill: def $al killed $al killed $eax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v8f64:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vcmpltpd %zmm0, %zmm1, %k1
; AVX512BW-NEXT:    vcmpltpd %zmm2, %zmm3, %k0 {%k1}
; AVX512BW-NEXT:    kmovd %k0, %eax
; AVX512BW-NEXT:    # kill: def $al killed $al killed $eax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = fcmp ogt <8 x double> %a, %b
  %x1 = fcmp ogt <8 x double> %c, %d
  %y = and <8 x i1> %x0, %x1
  %res = bitcast <8 x i1> %y to i8
  ret i8 %res
}

define i32 @v32i16(<32 x i16> %a, <32 x i16> %b, <32 x i16> %c, <32 x i16> %d) {
; SSE-LABEL: v32i16:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtw %xmm5, %xmm1
; SSE-NEXT:    pcmpgtw %xmm4, %xmm0
; SSE-NEXT:    packsswb %xmm1, %xmm0
; SSE-NEXT:    pcmpgtw %xmm7, %xmm3
; SSE-NEXT:    pcmpgtw %xmm6, %xmm2
; SSE-NEXT:    packsswb %xmm3, %xmm2
; SSE-NEXT:    pcmpgtw {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtw {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    packsswb %xmm11, %xmm10
; SSE-NEXT:    pand %xmm0, %xmm10
; SSE-NEXT:    pcmpgtw {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    pcmpgtw {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    packsswb %xmm9, %xmm8
; SSE-NEXT:    pand %xmm2, %xmm8
; SSE-NEXT:    pmovmskb %xmm10, %ecx
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    shll $16, %eax
; SSE-NEXT:    orl %ecx, %eax
; SSE-NEXT:    retq
;
; AVX1-LABEL: v32i16:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vextractf128 $1, %ymm3, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm9
; AVX1-NEXT:    vpcmpgtw %xmm8, %xmm9, %xmm8
; AVX1-NEXT:    vpcmpgtw %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpacksswb %xmm8, %xmm1, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpcmpgtw %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpcmpgtw %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpacksswb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vextractf128 $1, %ymm7, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm5, %xmm2
; AVX1-NEXT:    vpcmpgtw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpcmpgtw %xmm7, %xmm5, %xmm2
; AVX1-NEXT:    vpacksswb %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpand %xmm1, %xmm8, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm6, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm4, %xmm3
; AVX1-NEXT:    vpcmpgtw %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpcmpgtw %xmm6, %xmm4, %xmm3
; AVX1-NEXT:    vpacksswb %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpmovmskb %xmm0, %ecx
; AVX1-NEXT:    vpmovmskb %xmm1, %eax
; AVX1-NEXT:    shll $16, %eax
; AVX1-NEXT:    orl %ecx, %eax
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: v32i16:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpcmpgtw %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vpcmpgtw %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm2 = ymm0[2,3],ymm1[2,3]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    vpacksswb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpgtw %ymm7, %ymm5, %ymm1
; AVX2-NEXT:    vpcmpgtw %ymm6, %ymm4, %ymm2
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm3 = ymm2[2,3],ymm1[2,3]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; AVX2-NEXT:    vpacksswb %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpmovmskb %ymm0, %eax
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: v32i16:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vextracti64x4 $1, %zmm3, %ymm4
; AVX512F-NEXT:    vextracti64x4 $1, %zmm2, %ymm5
; AVX512F-NEXT:    vpcmpgtw %ymm4, %ymm5, %ymm4
; AVX512F-NEXT:    vextracti64x4 $1, %zmm1, %ymm5
; AVX512F-NEXT:    vextracti64x4 $1, %zmm0, %ymm6
; AVX512F-NEXT:    vpcmpgtw %ymm5, %ymm6, %ymm5
; AVX512F-NEXT:    vpand %ymm4, %ymm5, %ymm4
; AVX512F-NEXT:    vpcmpgtw %ymm1, %ymm0, %ymm0
; AVX512F-NEXT:    vpcmpgtw %ymm3, %ymm2, %ymm1
; AVX512F-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX512F-NEXT:    vpmovsxwd %ymm0, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %ecx
; AVX512F-NEXT:    vpmovsxwd %ymm4, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    shll $16, %eax
; AVX512F-NEXT:    orl %ecx, %eax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v32i16:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vpcmpgtw %zmm1, %zmm0, %k1
; AVX512BW-NEXT:    vpcmpgtw %zmm3, %zmm2, %k0 {%k1}
; AVX512BW-NEXT:    kmovd %k0, %eax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = icmp sgt <32 x i16> %a, %b
  %x1 = icmp sgt <32 x i16> %c, %d
  %y = and <32 x i1> %x0, %x1
  %res = bitcast <32 x i1> %y to i32
  ret i32 %res
}

define i16 @v16i32(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c, <16 x i32> %d) {
; SSE-LABEL: v16i32:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtd %xmm7, %xmm3
; SSE-NEXT:    pcmpgtd %xmm6, %xmm2
; SSE-NEXT:    packssdw %xmm3, %xmm2
; SSE-NEXT:    pcmpgtd %xmm5, %xmm1
; SSE-NEXT:    pcmpgtd %xmm4, %xmm0
; SSE-NEXT:    packssdw %xmm1, %xmm0
; SSE-NEXT:    packsswb %xmm2, %xmm0
; SSE-NEXT:    pcmpgtd {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtd {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    packssdw %xmm11, %xmm10
; SSE-NEXT:    pcmpgtd {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    pcmpgtd {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    packssdw %xmm9, %xmm8
; SSE-NEXT:    packsswb %xmm10, %xmm8
; SSE-NEXT:    pand %xmm0, %xmm8
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    # kill: def $ax killed $ax killed $eax
; SSE-NEXT:    retq
;
; AVX1-LABEL: v16i32:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vextractf128 $1, %ymm3, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm9
; AVX1-NEXT:    vpcmpgtd %xmm8, %xmm9, %xmm8
; AVX1-NEXT:    vpcmpgtd %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpackssdw %xmm8, %xmm1, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpcmpgtd %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpcmpgtd %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpackssdw %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpacksswb %xmm8, %xmm0, %xmm0
; AVX1-NEXT:    vextractf128 $1, %ymm7, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm5, %xmm2
; AVX1-NEXT:    vpcmpgtd %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpcmpgtd %xmm7, %xmm5, %xmm2
; AVX1-NEXT:    vpackssdw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm6, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm4, %xmm3
; AVX1-NEXT:    vpcmpgtd %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpcmpgtd %xmm6, %xmm4, %xmm3
; AVX1-NEXT:    vpackssdw %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpacksswb %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpmovmskb %xmm0, %eax
; AVX1-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: v16i32:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpcmpgtd %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm3
; AVX2-NEXT:    vpackssdw %xmm3, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpgtd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vpackssdw %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpacksswb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpgtd %ymm7, %ymm5, %ymm1
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpackssdw %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpgtd %ymm6, %ymm4, %ymm2
; AVX2-NEXT:    vextracti128 $1, %ymm2, %xmm3
; AVX2-NEXT:    vpackssdw %xmm3, %xmm2, %xmm2
; AVX2-NEXT:    vpacksswb %xmm1, %xmm2, %xmm1
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovmskb %xmm0, %eax
; AVX2-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: v16i32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vpcmpgtd %zmm1, %zmm0, %k1
; AVX512F-NEXT:    vpcmpgtd %zmm3, %zmm2, %k0 {%k1}
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v16i32:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vpcmpgtd %zmm1, %zmm0, %k1
; AVX512BW-NEXT:    vpcmpgtd %zmm3, %zmm2, %k0 {%k1}
; AVX512BW-NEXT:    kmovd %k0, %eax
; AVX512BW-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = icmp sgt <16 x i32> %a, %b
  %x1 = icmp sgt <16 x i32> %c, %d
  %y = and <16 x i1> %x0, %x1
  %res = bitcast <16 x i1> %y to i16
  ret i16 %res
}

define i16 @v16f32(<16 x float> %a, <16 x float> %b, <16 x float> %c, <16 x float> %d) {
; SSE-LABEL: v16f32:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    cmpltps %xmm3, %xmm7
; SSE-NEXT:    cmpltps %xmm2, %xmm6
; SSE-NEXT:    packssdw %xmm7, %xmm6
; SSE-NEXT:    cmpltps %xmm1, %xmm5
; SSE-NEXT:    cmpltps %xmm0, %xmm4
; SSE-NEXT:    packssdw %xmm5, %xmm4
; SSE-NEXT:    packsswb %xmm6, %xmm4
; SSE-NEXT:    cmpltps {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    cmpltps {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    packssdw %xmm11, %xmm10
; SSE-NEXT:    cmpltps {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    cmpltps {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    packssdw %xmm9, %xmm8
; SSE-NEXT:    packsswb %xmm10, %xmm8
; SSE-NEXT:    pand %xmm4, %xmm8
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    # kill: def $ax killed $ax killed $eax
; SSE-NEXT:    retq
;
; AVX12-LABEL: v16f32:
; AVX12:       # %bb.0:
; AVX12-NEXT:    vcmpltps %ymm1, %ymm3, %ymm1
; AVX12-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX12-NEXT:    vpackssdw %xmm3, %xmm1, %xmm1
; AVX12-NEXT:    vcmpltps %ymm0, %ymm2, %ymm0
; AVX12-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX12-NEXT:    vpackssdw %xmm2, %xmm0, %xmm0
; AVX12-NEXT:    vpacksswb %xmm1, %xmm0, %xmm0
; AVX12-NEXT:    vcmpltps %ymm5, %ymm7, %ymm1
; AVX12-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX12-NEXT:    vpackssdw %xmm2, %xmm1, %xmm1
; AVX12-NEXT:    vcmpltps %ymm4, %ymm6, %ymm2
; AVX12-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX12-NEXT:    vpackssdw %xmm3, %xmm2, %xmm2
; AVX12-NEXT:    vpacksswb %xmm1, %xmm2, %xmm1
; AVX12-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX12-NEXT:    vpmovmskb %xmm0, %eax
; AVX12-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX12-NEXT:    vzeroupper
; AVX12-NEXT:    retq
;
; AVX512F-LABEL: v16f32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vcmpltps %zmm0, %zmm1, %k1
; AVX512F-NEXT:    vcmpltps %zmm2, %zmm3, %k0 {%k1}
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v16f32:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vcmpltps %zmm0, %zmm1, %k1
; AVX512BW-NEXT:    vcmpltps %zmm2, %zmm3, %k0 {%k1}
; AVX512BW-NEXT:    kmovd %k0, %eax
; AVX512BW-NEXT:    # kill: def $ax killed $ax killed $eax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = fcmp ogt <16 x float> %a, %b
  %x1 = fcmp ogt <16 x float> %c, %d
  %y = and <16 x i1> %x0, %x1
  %res = bitcast <16 x i1> %y to i16
  ret i16 %res
}

define i64 @v64i8(<64 x i8> %a, <64 x i8> %b, <64 x i8> %c, <64 x i8> %d) {
; SSE-LABEL: v64i8:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pcmpgtb %xmm4, %xmm0
; SSE-NEXT:    pcmpgtb %xmm5, %xmm1
; SSE-NEXT:    pcmpgtb %xmm6, %xmm2
; SSE-NEXT:    pcmpgtb %xmm7, %xmm3
; SSE-NEXT:    pcmpgtb {{[0-9]+}}(%rsp), %xmm11
; SSE-NEXT:    pand %xmm0, %xmm11
; SSE-NEXT:    pcmpgtb {{[0-9]+}}(%rsp), %xmm10
; SSE-NEXT:    pand %xmm1, %xmm10
; SSE-NEXT:    pcmpgtb {{[0-9]+}}(%rsp), %xmm9
; SSE-NEXT:    pand %xmm2, %xmm9
; SSE-NEXT:    pcmpgtb {{[0-9]+}}(%rsp), %xmm8
; SSE-NEXT:    pand %xmm3, %xmm8
; SSE-NEXT:    pmovmskb %xmm11, %eax
; SSE-NEXT:    pmovmskb %xmm10, %ecx
; SSE-NEXT:    shll $16, %ecx
; SSE-NEXT:    orl %eax, %ecx
; SSE-NEXT:    pmovmskb %xmm9, %edx
; SSE-NEXT:    pmovmskb %xmm8, %eax
; SSE-NEXT:    shll $16, %eax
; SSE-NEXT:    orl %edx, %eax
; SSE-NEXT:    shlq $32, %rax
; SSE-NEXT:    orq %rcx, %rax
; SSE-NEXT:    retq
;
; AVX1-LABEL: v64i8:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vextractf128 $1, %ymm3, %xmm8
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm9
; AVX1-NEXT:    vpcmpgtb %xmm8, %xmm9, %xmm8
; AVX1-NEXT:    vpcmpgtb %xmm3, %xmm1, %xmm9
; AVX1-NEXT:    vextractf128 $1, %ymm2, %xmm3
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpcmpgtb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpcmpgtb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vextractf128 $1, %ymm7, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm5, %xmm3
; AVX1-NEXT:    vpcmpgtb %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpand %xmm2, %xmm8, %xmm2
; AVX1-NEXT:    vpcmpgtb %xmm7, %xmm5, %xmm3
; AVX1-NEXT:    vpand %xmm3, %xmm9, %xmm3
; AVX1-NEXT:    vextractf128 $1, %ymm6, %xmm5
; AVX1-NEXT:    vextractf128 $1, %ymm4, %xmm7
; AVX1-NEXT:    vpcmpgtb %xmm5, %xmm7, %xmm5
; AVX1-NEXT:    vpand %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpcmpgtb %xmm6, %xmm4, %xmm4
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpmovmskb %xmm0, %eax
; AVX1-NEXT:    vpmovmskb %xmm1, %ecx
; AVX1-NEXT:    shll $16, %ecx
; AVX1-NEXT:    orl %eax, %ecx
; AVX1-NEXT:    vpmovmskb %xmm3, %edx
; AVX1-NEXT:    vpmovmskb %xmm2, %eax
; AVX1-NEXT:    shll $16, %eax
; AVX1-NEXT:    orl %edx, %eax
; AVX1-NEXT:    shlq $32, %rax
; AVX1-NEXT:    orq %rcx, %rax
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: v64i8:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpcmpgtb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpgtb %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vpcmpgtb %ymm6, %ymm4, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpcmpgtb %ymm7, %ymm5, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpmovmskb %ymm0, %ecx
; AVX2-NEXT:    vpmovmskb %ymm1, %eax
; AVX2-NEXT:    shlq $32, %rax
; AVX2-NEXT:    orq %rcx, %rax
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: v64i8:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vextracti64x4 $1, %zmm3, %ymm4
; AVX512F-NEXT:    vextracti64x4 $1, %zmm2, %ymm5
; AVX512F-NEXT:    vpcmpgtb %ymm4, %ymm5, %ymm4
; AVX512F-NEXT:    vextracti64x4 $1, %zmm1, %ymm5
; AVX512F-NEXT:    vextracti64x4 $1, %zmm0, %ymm6
; AVX512F-NEXT:    vpcmpgtb %ymm5, %ymm6, %ymm5
; AVX512F-NEXT:    vextracti128 $1, %ymm5, %xmm6
; AVX512F-NEXT:    vpcmpgtb %ymm1, %ymm0, %ymm0
; AVX512F-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX512F-NEXT:    vextracti128 $1, %ymm4, %xmm7
; AVX512F-NEXT:    vpand %xmm7, %xmm6, %xmm6
; AVX512F-NEXT:    vpcmpgtb %ymm3, %ymm2, %ymm2
; AVX512F-NEXT:    vextracti128 $1, %ymm2, %xmm3
; AVX512F-NEXT:    vpand %xmm3, %xmm1, %xmm1
; AVX512F-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX512F-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    vpmovsxbd %xmm1, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %ecx
; AVX512F-NEXT:    shll $16, %ecx
; AVX512F-NEXT:    orl %eax, %ecx
; AVX512F-NEXT:    vpand %xmm4, %xmm5, %xmm0
; AVX512F-NEXT:    vpmovsxbd %xmm0, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %edx
; AVX512F-NEXT:    vpmovsxbd %xmm6, %zmm0
; AVX512F-NEXT:    vptestmd %zmm0, %zmm0, %k0
; AVX512F-NEXT:    kmovw %k0, %eax
; AVX512F-NEXT:    shll $16, %eax
; AVX512F-NEXT:    orl %edx, %eax
; AVX512F-NEXT:    shlq $32, %rax
; AVX512F-NEXT:    orq %rcx, %rax
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512BW-LABEL: v64i8:
; AVX512BW:       # %bb.0:
; AVX512BW-NEXT:    vpcmpgtb %zmm1, %zmm0, %k1
; AVX512BW-NEXT:    vpcmpgtb %zmm3, %zmm2, %k0 {%k1}
; AVX512BW-NEXT:    kmovq %k0, %rax
; AVX512BW-NEXT:    vzeroupper
; AVX512BW-NEXT:    retq
  %x0 = icmp sgt <64 x i8> %a, %b
  %x1 = icmp sgt <64 x i8> %c, %d
  %y = and <64 x i1> %x0, %x1
  %res = bitcast <64 x i1> %y to i64
  ret i64 %res
}
