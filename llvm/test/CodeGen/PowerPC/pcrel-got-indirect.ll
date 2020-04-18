; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -enable-ppc-quad-precision -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s

%struct.Struct = type { i8, i16, i32 }

@valChar = external local_unnamed_addr global i8, align 1
@valShort = external local_unnamed_addr global i16, align 2
@valInt = external global i32, align 4
@valUnsigned = external local_unnamed_addr global i32, align 4
@valLong = external local_unnamed_addr global i64, align 8
@ptr = external local_unnamed_addr global i32*, align 8
@array = external local_unnamed_addr global [10 x i32], align 4
@structure = external local_unnamed_addr global %struct.Struct, align 4
@ptrfunc = external local_unnamed_addr global void (...)*, align 8

define dso_local signext i32 @ReadGlobalVarChar() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalVarChar:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valChar@got@pcrel(0), 1
; CHECK-NEXT:    lbz r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i8, i8* @valChar, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define dso_local void @WriteGlobalVarChar() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalVarChar:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valChar@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 3
; CHECK-NEXT:    stb r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store i8 3, i8* @valChar, align 1
  ret void
}

define dso_local signext i32 @ReadGlobalVarShort() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalVarShort:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valShort@got@pcrel(0), 1
; CHECK-NEXT:    lha r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i16, i16* @valShort, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

define dso_local void @WriteGlobalVarShort() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalVarShort:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valShort@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 3
; CHECK-NEXT:    sth r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store i16 3, i16* @valShort, align 2
  ret void
}

define dso_local signext i32 @ReadGlobalVarInt() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalVarInt:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valInt@got@pcrel(0), 1
; CHECK-NEXT:    lwa r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32, i32* @valInt, align 4
  ret i32 %0
}

define dso_local void @WriteGlobalVarInt() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalVarInt:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valInt@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 33
; CHECK-NEXT:    stw r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store i32 33, i32* @valInt, align 4
  ret void
}

define dso_local signext i32 @ReadGlobalVarUnsigned() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalVarUnsigned:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valUnsigned@got@pcrel(0), 1
; CHECK-NEXT:    lwa r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32, i32* @valUnsigned, align 4
  ret i32 %0
}

define dso_local void @WriteGlobalVarUnsigned() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalVarUnsigned:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valUnsigned@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 33
; CHECK-NEXT:    stw r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store i32 33, i32* @valUnsigned, align 4
  ret void
}

define dso_local signext i32 @ReadGlobalVarLong() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalVarLong:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valLong@got@pcrel(0), 1
; CHECK-NEXT:    lwa r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i64, i64* @valLong, align 8
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define dso_local void @WriteGlobalVarLong() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalVarLong:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valLong@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 3333
; CHECK-NEXT:    std r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store i64 3333, i64* @valLong, align 8
  ret void
}

define dso_local i32* @ReadGlobalPtr() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, ptr@got@pcrel(0), 1
; CHECK-NEXT:    ld r3, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32*, i32** @ptr, align 8
  ret i32* %0
}

define dso_local void @WriteGlobalPtr() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, ptr@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 3
; CHECK-NEXT:    ld r3, 0(r3)
; CHECK-NEXT:    stw r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32*, i32** @ptr, align 8
  store i32 3, i32* %0, align 4
  ret void
}

define dso_local nonnull i32* @GlobalVarAddr() local_unnamed_addr  {
; CHECK-LABEL: GlobalVarAddr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, valInt@got@pcrel(0), 1
; CHECK-NEXT:    blr
entry:
  ret i32* @valInt
}

define dso_local signext i32 @ReadGlobalArray() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalArray:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, array@got@pcrel(0), 1
; CHECK-NEXT:    lwa r3, 12(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array, i64 0, i64 3), align 4
  ret i32 %0
}

define dso_local void @WriteGlobalArray() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalArray:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, array@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 5
; CHECK-NEXT:    stw r4, 12(r3)
; CHECK-NEXT:    blr
entry:
  store i32 5, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array, i64 0, i64 3), align 4
  ret void
}

define dso_local signext i32 @ReadGlobalStruct() local_unnamed_addr  {
; CHECK-LABEL: ReadGlobalStruct:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, structure@got@pcrel(0), 1
; CHECK-NEXT:    lwa r3, 4(r3)
; CHECK-NEXT:    blr
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.Struct, %struct.Struct* @structure, i64 0, i32 2), align 4
  ret i32 %0
}

define dso_local void @WriteGlobalStruct() local_unnamed_addr  {
; CHECK-LABEL: WriteGlobalStruct:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, structure@got@pcrel(0), 1
; CHECK-NEXT:    li r4, 3
; CHECK-NEXT:    stw r4, 4(r3)
; CHECK-NEXT:    blr
entry:
  store i32 3, i32* getelementptr inbounds (%struct.Struct, %struct.Struct* @structure, i64 0, i32 2), align 4
  ret void
}

define dso_local void @ReadFuncPtr() local_unnamed_addr  {
; CHECK-LABEL: ReadFuncPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    std r0, 16(r1)
; CHECK-NEXT:    stdu r1, -32(r1)
; CHECK-NEXT:    std r2, 24(r1)
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    .cfi_offset lr, 16
; CHECK-NEXT:    pld r3, ptrfunc@got@pcrel(0), 1
; CHECK-NEXT:    ld r12, 0(r3)
; CHECK-NEXT:    mtctr r12
; CHECK-NEXT:    bctrl
; CHECK-NEXT:    ld 2, 24(r1)
; CHECK-NEXT:    addi r1, r1, 32
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
entry:
  %0 = load void ()*, void ()** bitcast (void (...)** @ptrfunc to void ()**), align 8
  tail call void %0()
  ret void
}

define dso_local void @WriteFuncPtr() local_unnamed_addr  {
; CHECK-LABEL: WriteFuncPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pld r3, ptrfunc@got@pcrel(0), 1
; CHECK-NEXT:    pld r4, function@got@pcrel(0), 1
; CHECK-NEXT:    std r4, 0(r3)
; CHECK-NEXT:    blr
entry:
  store void (...)* @function, void (...)** @ptrfunc, align 8
  ret void
}

declare void @function(...)

