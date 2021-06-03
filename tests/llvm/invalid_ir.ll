; Generated using llvm-stress --seed=1068
; This IR file can be assembled: $ lvm-as tests/llvm/invalid_ir.ll
; But it cannot be compiled:     $ clang tests/llvm/invalid_ir.ll
; The error is: "error in backend: Cannot emit physreg copy instruction"
;
; Copyright (c) Facebook, Inc. and its affiliates.
;
; This source code is licensed under the MIT license found in the
; LICENSE file in the root directory of this source tree.

; ModuleID = '<stdin>'
source_filename = "/tmp/autogen.bc"

define void @autogen_SD1068(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
BB:
  %A4 = alloca <8 x i32>
  %A3 = alloca <16 x double>
  %A2 = alloca <8 x double>
  %A1 = alloca <1 x float>
  %A = alloca <1 x i64>
  %L = load i8, i8* %0
  store <1 x i64> <i64 -1>, <1 x i64>* %A
  %E = extractelement <16 x i32> zeroinitializer, i32 7
  %Shuff = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 undef, i32 0>
  %I = insertelement <2 x i1> zeroinitializer, i1 true, i32 0
  %B = srem i8 %5, %5
  %FC = uitofp <4 x i1> zeroinitializer to <4 x double>
  %Sl = select i1 false, i16 -83, i16 -1
  %Cmp = icmp slt <16 x i32> zeroinitializer, zeroinitializer
  %L5 = load i8, i8* %0
  store i8 -11, i8* %0
  %E6 = extractelement <2 x i1> zeroinitializer, i32 1
  br label %CF74

CF74:                                             ; preds = %BB
  %Shuff7 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff, <2 x i32> <i32 undef, i32 2>
  %I8 = insertelement <2 x i1> %Shuff, i1 false, i32 0
  %B9 = and i16 %Sl, -25375
  %FC10 = uitofp <2 x i1> zeroinitializer to <2 x float>
  %Sl11 = select i1 false, i16 -25375, i16 -1
  %Cmp12 = icmp uge <2 x i1> %I, %Shuff
  %L13 = load i8, i8* %0
  store i8 -111, i8* %0
  %E14 = extractelement <4 x double> %FC, i32 3
  %Shuff15 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff, <2 x i32> <i32 2, i32 0>
  %I16 = insertelement <2 x float> %FC10, float 0xB8A8325C00000000, i32 0
  %B17 = srem <2 x i64> zeroinitializer, zeroinitializer
  %PC = bitcast <8 x i32>* %A4 to i1*
  %Sl18 = select i1 false, <2 x float> %FC10, <2 x float> %FC10
  %Cmp19 = icmp ule <8 x i16> zeroinitializer, zeroinitializer
  %L20 = load i1, i1* %PC
  br label %CF

CF:                                               ; preds = %CF76, %CF82, %CF, %CF74
  store i1 true, i1* %PC
  %E21 = extractelement <2 x i1> zeroinitializer, i32 1
  br i1 %E21, label %CF, label %CF77

CF77:                                             ; preds = %CF83, %CF77, %CF
  %Shuff22 = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %I23 = insertelement <2 x i16> zeroinitializer, i16 -83, i32 0
  %B24 = shl <2 x i16> zeroinitializer, zeroinitializer
  %FC25 = fptosi float 0x3C2A305E00000000 to i32
  %Sl26 = select i1 true, <8 x i8> zeroinitializer, <8 x i8> zeroinitializer
  %L27 = load i1, i1* %PC
  br i1 %L27, label %CF77, label %CF83

CF83:                                             ; preds = %CF77
  store i32 %FC25, i32* %1
  %E28 = extractelement <4 x double> %FC, i32 0
  %Shuff29 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff, <2 x i32> <i32 3, i32 1>
  %I30 = insertelement <16 x i32> zeroinitializer, i32 484009, i32 1
  %B31 = fdiv float 0x3C2A305E00000000, 0x401A204E00000000
  %Sl32 = select i1 false, double %E14, double 0xE8B721D6BDB52C54
  %Cmp33 = fcmp une float 0xB8A8325C00000000, %B31
  br i1 %Cmp33, label %CF77, label %CF79

CF79:                                             ; preds = %CF79, %CF83
  %L34 = load i1, i1* %PC
  br i1 %L34, label %CF79, label %CF80

CF80:                                             ; preds = %CF80, %CF79
  store i1 false, i1* %PC
  %E35 = extractelement <8 x i1> %Cmp19, i32 1
  br i1 %E35, label %CF80, label %CF82

CF82:                                             ; preds = %CF80
  %Shuff36 = shufflevector <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> <i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 undef, i32 18, i32 20, i32 22, i32 undef, i32 26, i32 28, i32 30, i32 0, i32 undef>
  %I37 = insertelement <16 x i32> zeroinitializer, i32 48253, i32 6
  %ZE = zext <2 x i1> %Shuff29 to <2 x i64>
  %Sl38 = select i1 true, i1 false, i1 false
  br i1 %Sl38, label %CF, label %CF75

CF75:                                             ; preds = %CF81, %CF75, %CF82
  %L39 = load i8, i8* %0
  store i1 true, i1* %PC
  %E40 = extractelement <16 x i32> zeroinitializer, i32 9
  %Shuff41 = shufflevector <2 x i1> %Shuff29, <2 x i1> %Shuff29, <2 x i32> <i32 0, i32 2>
  %I42 = insertelement <2 x i1> zeroinitializer, i1 false, i32 0
  %Tr = trunc <16 x i32> %I37 to <16 x i16>
  %Sl43 = select i1 %Sl38, i16 %Sl, i16 %Sl
  %Cmp44 = icmp sge <2 x i1> zeroinitializer, zeroinitializer
  %L45 = load i8, i8* %0
  store i8 %5, i8* %0
  %E46 = extractelement <2 x i16> %B24, i32 1
  %Shuff47 = shufflevector <2 x i1> %Shuff41, <2 x i1> %Shuff, <2 x i32> <i32 0, i32 2>
  %I48 = insertelement <2 x i1> zeroinitializer, i1 %E6, i32 0
  %B49 = shl i32 48253, %FC25
  %FC50 = uitofp <2 x i1> %Cmp12 to <2 x double>
  %Sl51 = select i1 %E6, <16 x i32> %I30, <16 x i32> zeroinitializer
  %Cmp52 = fcmp ult double 0x840B362A8B09F2A8, %E14
  br i1 %Cmp52, label %CF75, label %CF78

CF78:                                             ; preds = %CF78, %CF75
  %L53 = load i8, i8* %0
  store i8 -79, i8* %0
  %E54 = extractelement <8 x i16> zeroinitializer, i32 4
  %Shuff55 = shufflevector <2 x i1> %Shuff, <2 x i1> %Shuff, <2 x i32> <i32 3, i32 undef>
  %I56 = insertelement <2 x i1> %I8, i1 %Cmp52, i32 1
  %Sl57 = select i1 %Cmp52, i8 %L13, i8 %L39
  %Cmp58 = icmp ne <2 x i64> zeroinitializer, %B17
  %L59 = load i1, i1* %PC
  br i1 %L59, label %CF78, label %CF81

CF81:                                             ; preds = %CF78
  store i8 -111, i8* %0
  %E60 = extractelement <2 x i1> %Shuff7, i32 1
  br i1 %E60, label %CF75, label %CF76

CF76:                                             ; preds = %CF81
  %Shuff61 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff47, <2 x i32> <i32 2, i32 0>
  %I62 = insertelement <2 x i1> zeroinitializer, i1 %E21, i32 0
  %B63 = frem double %Sl32, %E14
  %ZE64 = zext i1 false to i8
  %Sl65 = select i1 false, <2 x i1> %Shuff61, <2 x i1> %Shuff
  %Cmp66 = icmp uge i64 %4, %4
  br i1 %Cmp66, label %CF, label %CF73

CF73:                                             ; preds = %CF76
  %L67 = load i8, i8* %0
  store i8 -111, i8* %0
  %E68 = extractelement <2 x i16> %B24, i32 1
  %Shuff69 = shufflevector <16 x i32> zeroinitializer, <16 x i32> %I30, <16 x i32> <i32 14, i32 16, i32 undef, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12>
  %I70 = insertelement <16 x i32> zeroinitializer, i32 %B49, i32 0
  %Tr71 = fptrunc double 0x840B362A8B09F2A8 to float
  %Sl72 = select i1 %Cmp52, i32 %FC25, i32 48253
  store i8 %L39, i8* %0
  store i8 %L13, i8* %0
  store i1 %Cmp52, i1* %PC
  store <1 x i64> <i64 -1>, <1 x i64>* %A
  store i8 %L39, i8* %0
  ret void
}
