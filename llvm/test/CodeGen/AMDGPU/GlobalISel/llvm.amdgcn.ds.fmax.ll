; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; NOTE: Assertions have been autogenerated by utils/update_mir_test_checks.py
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GFX8 %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GFX9 %s

; Make sure the memory operand information is preserved.
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=tonga -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GFX8-MIR %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GFX9-MIR %s


define amdgpu_ps float @ds_fmax_f32_ss(float addrspace(3)* inreg %ptr, float inreg %val) {
; GFX8-LABEL: ds_fmax_f32_ss:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    v_mov_b32_e32 v0, s2
; GFX8-NEXT:    v_mov_b32_e32 v1, s3
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    ; return to shader part epilog
;
; GFX9-LABEL: ds_fmax_f32_ss:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    v_mov_b32_e32 v0, s2
; GFX9-NEXT:    v_mov_b32_e32 v1, s3
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_ss
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[COPY]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY2]], [[COPY3]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX8-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_]]
  ; GFX8-MIR:   SI_RETURN_TO_EPILOG implicit $vgpr0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_ss
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[COPY]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY2]], [[COPY3]], 0, 0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX9-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_gfx9_]]
  ; GFX9-MIR:   SI_RETURN_TO_EPILOG implicit $vgpr0
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %ptr, float %val, i32 0, i32 0, i1 false)
  ret float %ret
}

define amdgpu_ps float @ds_fmax_f32_ss_offset(float addrspace(3)* inreg %ptr, float inreg %val) {
; GFX8-LABEL: ds_fmax_f32_ss_offset:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_add_u32 s0, s2, 0x200
; GFX8-NEXT:    v_mov_b32_e32 v0, s0
; GFX8-NEXT:    v_mov_b32_e32 v1, s3
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    ; return to shader part epilog
;
; GFX9-LABEL: ds_fmax_f32_ss_offset:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_add_u32 s0, s2, 0x200
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s3
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_ss_offset
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX8-MIR:   [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 512
  ; GFX8-MIR:   [[S_ADD_U32_:%[0-9]+]]:sreg_32 = S_ADD_U32 [[COPY]], [[S_MOV_B32_]], implicit-def $scc
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[S_ADD_U32_]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY2]], [[COPY3]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX8-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_]]
  ; GFX8-MIR:   SI_RETURN_TO_EPILOG implicit $vgpr0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_ss_offset
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX9-MIR:   [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 512
  ; GFX9-MIR:   [[S_ADD_U32_:%[0-9]+]]:sreg_32 = S_ADD_U32 [[COPY]], [[S_MOV_B32_]], implicit-def $scc
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[S_ADD_U32_]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY2]], [[COPY3]], 0, 0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX9-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_gfx9_]]
  ; GFX9-MIR:   SI_RETURN_TO_EPILOG implicit $vgpr0
  %gep = getelementptr float, float addrspace(3)* %ptr, i32 128
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %gep, float %val, i32 0, i32 0, i1 false)
  ret float %ret
}

define amdgpu_ps void @ds_fmax_f32_ss_nortn(float addrspace(3)* inreg %ptr, float inreg %val) {
; GFX8-LABEL: ds_fmax_f32_ss_nortn:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    v_mov_b32_e32 v0, s2
; GFX8-NEXT:    v_mov_b32_e32 v1, s3
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_endpgm
;
; GFX9-LABEL: ds_fmax_f32_ss_nortn:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    v_mov_b32_e32 v0, s2
; GFX9-NEXT:    v_mov_b32_e32 v1, s3
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_endpgm
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_ss_nortn
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[COPY]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY2]], [[COPY3]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX8-MIR:   S_ENDPGM 0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_ss_nortn
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[COPY]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY2]], [[COPY3]], 0, 0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX9-MIR:   S_ENDPGM 0
  %unused = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %ptr, float %val, i32 0, i32 0, i1 false)
  ret void
}

define amdgpu_ps void @ds_fmax_f32_ss_offset_nortn(float addrspace(3)* inreg %ptr, float inreg %val) {
; GFX8-LABEL: ds_fmax_f32_ss_offset_nortn:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_add_u32 s0, s2, 0x200
; GFX8-NEXT:    v_mov_b32_e32 v0, s0
; GFX8-NEXT:    v_mov_b32_e32 v1, s3
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_endpgm
;
; GFX9-LABEL: ds_fmax_f32_ss_offset_nortn:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_add_u32 s0, s2, 0x200
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s3
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_endpgm
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_ss_offset_nortn
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX8-MIR:   [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 512
  ; GFX8-MIR:   [[S_ADD_U32_:%[0-9]+]]:sreg_32 = S_ADD_U32 [[COPY]], [[S_MOV_B32_]], implicit-def $scc
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[S_ADD_U32_]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY2]], [[COPY3]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX8-MIR:   S_ENDPGM 0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_ss_offset_nortn
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $sgpr2, $sgpr3
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:sreg_32 = COPY $sgpr2
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:sreg_32 = COPY $sgpr3
  ; GFX9-MIR:   [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 512
  ; GFX9-MIR:   [[S_ADD_U32_:%[0-9]+]]:sreg_32 = S_ADD_U32 [[COPY]], [[S_MOV_B32_]], implicit-def $scc
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY [[S_ADD_U32_]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY [[COPY1]]
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY2]], [[COPY3]], 0, 0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX9-MIR:   S_ENDPGM 0
  %gep = getelementptr float, float addrspace(3)* %ptr, i32 128
  %unused = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %gep, float %val, i32 0, i32 0, i1 false)
  ret void
}

define float @ds_fmax_f32_vv(float addrspace(3)* %ptr, float %val) {
; GFX8-LABEL: ds_fmax_f32_vv:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: ds_fmax_f32_vv:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_vv
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY]], [[COPY1]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX8-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX8-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_vv
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY]], [[COPY1]], 0, 0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX9-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_gfx9_]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX9-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %ptr, float %val, i32 0, i32 0, i1 false)
  ret float %ret
}

define float @ds_fmax_f32_vv_offset(float addrspace(3)* %ptr, float %val) {
; GFX8-LABEL: ds_fmax_f32_vv_offset:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1 offset:512
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: ds_fmax_f32_vv_offset:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1 offset:512
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_vv_offset
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY]], [[COPY1]], 512, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX8-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX8-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_vv_offset
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY]], [[COPY1]], 512, 0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX9-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_gfx9_]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX9-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  %gep = getelementptr float, float addrspace(3)* %ptr, i32 128
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %gep, float %val, i32 0, i32 0, i1 false)
  ret float %ret
}

define void @ds_fmax_f32_vv_nortn(float addrspace(3)* %ptr, float %val) {
; GFX8-LABEL: ds_fmax_f32_vv_nortn:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: ds_fmax_f32_vv_nortn:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_vv_nortn
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY]], [[COPY1]], 0, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX8-MIR:   S_SETPC_B64_return [[COPY3]]
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_vv_nortn
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY]], [[COPY1]], 0, 0, implicit $exec :: (load store 4 on %ir.ptr, addrspace 3)
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX9-MIR:   S_SETPC_B64_return [[COPY3]]
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %ptr, float %val, i32 0, i32 0, i1 false)
  ret void
}

define void @ds_fmax_f32_vv_offset_nortn(float addrspace(3)* %ptr, float %val) {
; GFX8-LABEL: ds_fmax_f32_vv_offset_nortn:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1 offset:512
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: ds_fmax_f32_vv_offset_nortn:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1 offset:512
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_vv_offset_nortn
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY]], [[COPY1]], 512, 0, implicit $m0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX8-MIR:   S_SETPC_B64_return [[COPY3]]
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_vv_offset_nortn
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY]], [[COPY1]], 512, 0, implicit $exec :: (load store 4 on %ir.gep, addrspace 3)
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX9-MIR:   S_SETPC_B64_return [[COPY3]]
  %gep = getelementptr float, float addrspace(3)* %ptr, i32 128
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %gep, float %val, i32 0, i32 0, i1 false)
  ret void
}

define float @ds_fmax_f32_vv_volatile(float addrspace(3)* %ptr, float %val) {
; GFX8-LABEL: ds_fmax_f32_vv_volatile:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 m0, -1
; GFX8-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: ds_fmax_f32_vv_volatile:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_max_rtn_f32 v0, v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  ; GFX8-MIR-LABEL: name: ds_fmax_f32_vv_volatile
  ; GFX8-MIR: bb.1 (%ir-block.0):
  ; GFX8-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX8-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX8-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX8-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX8-MIR:   $m0 = S_MOV_B32 -1
  ; GFX8-MIR:   [[DS_MAX_RTN_F32_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32 [[COPY]], [[COPY1]], 0, 0, implicit $m0, implicit $exec :: (volatile load store 4 on %ir.ptr, addrspace 3)
  ; GFX8-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_]]
  ; GFX8-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX8-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  ; GFX9-MIR-LABEL: name: ds_fmax_f32_vv_volatile
  ; GFX9-MIR: bb.1 (%ir-block.0):
  ; GFX9-MIR:   liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31
  ; GFX9-MIR:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; GFX9-MIR:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; GFX9-MIR:   [[COPY2:%[0-9]+]]:sgpr_64 = COPY $sgpr30_sgpr31
  ; GFX9-MIR:   [[DS_MAX_RTN_F32_gfx9_:%[0-9]+]]:vgpr_32 = DS_MAX_RTN_F32_gfx9 [[COPY]], [[COPY1]], 0, 0, implicit $exec :: (volatile load store 4 on %ir.ptr, addrspace 3)
  ; GFX9-MIR:   $vgpr0 = COPY [[DS_MAX_RTN_F32_gfx9_]]
  ; GFX9-MIR:   [[COPY3:%[0-9]+]]:ccr_sgpr_64 = COPY [[COPY2]]
  ; GFX9-MIR:   S_SETPC_B64_return [[COPY3]], implicit $vgpr0
  %ret = call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %ptr, float %val, i32 0, i32 0, i1 true)
  ret float %ret
}

declare float @llvm.amdgcn.ds.fmax(float addrspace(3)* nocapture, float, i32 immarg, i32 immarg, i1 immarg) #0

attributes #0 = { argmemonly nounwind willreturn }
