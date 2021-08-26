/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const instCountDict = [
  {
    Index: 0,
    Name: "TotalInsts",
    Description: "Total instruction count",
  },
  {
    Index: 1,
    Name: "TotalBlocks",
    Description: "Basic block count",
  },
  {
    Index: 2,
    Name: "TotalFuncs",
    Description: "Function count",
  },
  {
    Index: 3,
    Name: "Ret",
    Description: "Ret instruction count",
  },
  {
    Index: 4,
    Name: "Br",
    Description: "Br instruction count",
  },
  {
    Index: 5,
    Name: "Switch",
    Description: "Switch instruction count",
  },
  {
    Index: 6,
    Name: "IndirectBr",
    Description: "IndirectBr instruction count",
  },
  {
    Index: 7,
    Name: "Invoke",
    Description: "Invoke instruction count",
  },
  {
    Index: 8,
    Name: "Resume",
    Description: "Resume instruction count",
  },
  {
    Index: 9,
    Name: "Unreachable",
    Description: "Unreachable instruction count",
  },
  {
    Index: 10,
    Name: "CleanupRet",
    Description: "CleanupRet instruction count",
  },
  {
    Index: 11,
    Name: "CatchRet",
    Description: "CatchRet instruction count",
  },
  {
    Index: 12,
    Name: "CatchSwitch",
    Description: "CatchSwitch instruction count",
  },
  {
    Index: 13,
    Name: "CallBr",
    Description: "CallBr instruction count",
  },
  {
    Index: 14,
    Name: "FNeg",
    Description: "FNeg instruction count",
  },
  {
    Index: 15,
    Name: "Add",
    Description: "Add instruction count",
  },
  {
    Index: 16,
    Name: "FAdd",
    Description: "FAdd instruction count",
  },
  {
    Index: 17,
    Name: "Sub",
    Description: "Sub instruction count",
  },
  {
    Index: 18,
    Name: "FSub",
    Description: "FSub instruction count",
  },
  {
    Index: 19,
    Name: "Mul",
    Description: "Mul instruction count",
  },
  {
    Index: 20,
    Name: "FMul",
    Description: "FMul instruction count",
  },
  {
    Index: 21,
    Name: "UDiv",
    Description: "UDiv instruction count",
  },
  {
    Index: 22,
    Name: "SDiv",
    Description: "SDiv instruction count",
  },
  {
    Index: 23,
    Name: "FDiv",
    Description: "FDiv instruction count",
  },
  {
    Index: 24,
    Name: "URem",
    Description: "URem instruction count",
  },
  {
    Index: 25,
    Name: "SRem",
    Description: "SRem instruction count",
  },
  {
    Index: 26,
    Name: "FRem",
    Description: "FRem instruction count",
  },
  {
    Index: 27,
    Name: "Shl",
    Description: "Shl instruction count",
  },
  {
    Index: 28,
    Name: "LShr",
    Description: "LShr instruction count",
  },
  {
    Index: 29,
    Name: "AShr",
    Description: "AShr instruction count",
  },
  {
    Index: 30,
    Name: "And",
    Description: "And instruction count",
  },
  {
    Index: 31,
    Name: "Or",
    Description: "Or instruction count",
  },
  {
    Index: 32,
    Name: "Xor",
    Description: "Xor instruction count",
  },
  {
    Index: 33,
    Name: "Alloca",
    Description: "Alloca instruction count",
  },
  {
    Index: 34,
    Name: "Load",
    Description: "Load instruction count",
  },
  {
    Index: 35,
    Name: "Store",
    Description: "Store instruction count",
  },
  {
    Index: 36,
    Name: "GetElementPtr",
    Description: "GetElementPtr instruction count",
  },
  {
    Index: 37,
    Name: "Fence",
    Description: "Fence instruction count",
  },
  {
    Index: 38,
    Name: "AtomicCmpXchg",
    Description: "AtomicCmpXchg instruction count",
  },
  {
    Index: 39,
    Name: "AtomicRMW",
    Description: "AtomicRMW instruction count",
  },
  {
    Index: 40,
    Name: "Trunc",
    Description: "Trunc instruction count",
  },
  {
    Index: 41,
    Name: "ZExt",
    Description: "ZExt instruction count",
  },
  {
    Index: 42,
    Name: "SExt",
    Description: "SExt instruction count",
  },
  {
    Index: 43,
    Name: "FPToUI",
    Description: "FPToUI instruction count",
  },
  {
    Index: 44,
    Name: "FPToSI",
    Description: "FPToSI instruction count",
  },
  {
    Index: 45,
    Name: "UIToFP",
    Description: "UIToFP instruction count",
  },
  {
    Index: 46,
    Name: "SIToFP",
    Description: "SIToFP instruction count",
  },
  {
    Index: 47,
    Name: "FPTrunc",
    Description: "FPTrunc instruction count",
  },
  {
    Index: 48,
    Name: "FPExt",
    Description: "FPExt instruction count",
  },
  {
    Index: 49,
    Name: "PtrToInt",
    Description: "PtrToInt instruction count",
  },
  {
    Index: 50,
    Name: "IntToPtr",
    Description: "IntToPtr instruction count",
  },
  {
    Index: 51,
    Name: "BitCast",
    Description: "BitCast instruction count",
  },
  {
    Index: 52,
    Name: "AddrSpaceCast",
    Description: "AddrSpaceCast instruction count",
  },
  {
    Index: 53,
    Name: "CleanupPad",
    Description: "CleanupPad instruction count",
  },
  {
    Index: 54,
    Name: "CatchPad",
    Description: "CatchPad instruction count",
  },
  {
    Index: 55,
    Name: "ICmp",
    Description: "ICmp instruction count",
  },
  {
    Index: 56,
    Name: "FCmp",
    Description: "FCmp instruction count",
  },
  {
    Index: 57,
    Name: "PHI",
    Description: "PHI instruction count",
  },
  {
    Index: 58,
    Name: "Call",
    Description: "Call instruction count",
  },
  {
    Index: 59,
    Name: "Select",
    Description: "Select instruction count",
  },
  {
    Index: 60,
    Name: "UserOp1",
    Description: "UserOp1 instruction count",
  },
  {
    Index: 61,
    Name: "UserOp2",
    Description: "UserOp2 instruction count",
  },
  {
    Index: 62,
    Name: "VAArg",
    Description: "VAArg instruction count",
  },
  {
    Index: 63,
    Name: "ExtractElement",
    Description: "ExtractElement instruction count",
  },
  {
    Index: 64,
    Name: "InsertElement",
    Description: "InsertElement instruction count",
  },
  {
    Index: 65,
    Name: "ShuffleVector",
    Description: "ShuffleVector instruction count",
  },
  {
    Index: 66,
    Name: "ExtractValue",
    Description: "ExtractValue instruction count",
  },
  {
    Index: 67,
    Name: "InsertValue",
    Description: "InsertValue instruction count",
  },
  {
    Index: 68,
    Name: "LandingPad",
    Description: "LandingPad instruction count",
  },
  {
    Index: 69,
    Name: "Freeze",
    Description: "Freeze instruction count",
  },
];

export default instCountDict;
