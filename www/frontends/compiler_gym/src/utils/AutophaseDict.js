const autophaseDict = [
  {
    Index: 0,
    Name: "BBNumArgsHi",
    Description: "Number of BB where total args for phi nodes is gt 5",
  },
  {
    Index: 1,
    Name: "BBNumArgsLo",
    Description: "Number of BB where total args for phi nodes is [1, 5]",
  },
  {
    Index: 2,
    Name: "onePred",
    Description: "Number of basic blocks with 1 predecessor",
  },
  {
    Index: 3,
    Name: "onePredOneSuc",
    Description: "Number of basic blocks with 1 predecessor and 1 successor",
  },
  {
    Index: 4,
    Name: "onePredTwoSuc",
    Description: "Number of basic blocks with 1 predecessor and 2 successors",
  },
  {
    Index: 5,
    Name: "oneSuccessor",
    Description: "Number of basic blocks with 1 successor",
  },
  {
    Index: 6,
    Name: "twoPred",
    Description: "Number of basic blocks with 2 predecessors",
  },
  {
    Index: 7,
    Name: "twoPredOneSuc",
    Description: "Number of basic blocks with 2 predecessors and 1 successor",
  },
  {
    Index: 8,
    Name: "twoEach",
    Description: "Number of basic blocks with 2 predecessors and successors",
  },
  {
    Index: 9,
    Name: "twoSuccessor",
    Description: "Number of basic blocks with 2 successors",
  },
  {
    Index: 10,
    Name: "morePreds",
    Description: "Number of basic blocks with gt. 2 predecessors",
  },
  {
    Index: 11,
    Name: "BB03Phi",
    Description: "Number of basic blocks with Phi node count in range (0, 3]",
  },
  {
    Index: 12,
    Name: "BBHiPhi",
    Description: "Number of basic blocks with more than 3 Phi nodes",
  },
  {
    Index: 13,
    Name: "BBNoPhi",
    Description: "Number of basic blocks with no Phi nodes",
  },
  {
    Index: 14,
    Name: "BeginPhi",
    Description: "Number of Phi-nodes at beginning of BB",
  },
  {
    Index: 15,
    Name: "BranchCount",
    Description: "Number of branches",
  },
  {
    Index: 16,
    Name: "returnInt",
    Description: "Number of calls that return an int",
  },
  {
    Index: 17,
    Name: "CriticalCount",
    Description: "Number of critical edges",
  },
  {
    Index: 18,
    Name: "NumEdges",
    Description: "Number of edges",
  },
  {
    Index: 19,
    Name: "const32Bit",
    Description: "Number of occurrences of 32-bit integer constants",
  },
  {
    Index: 20,
    Name: "const64Bit",
    Description: "Number of occurrences of 64-bit integer constants",
  },
  {
    Index: 21,
    Name: "numConstZeroes",
    Description: "Number of occurrences of constant 0",
  },
  {
    Index: 22,
    Name: "numConstOnes",
    Description: "Number of occurrences of constant 1",
  },
  {
    Index: 23,
    Name: "UncondBranches",
    Description: "Number of unconditional branches",
  },
  {
    Index: 24,
    Name: "binaryConstArg",
    Description: "Binary operations with a constant operand",
  },
  {
    Index: 25,
    Name: "NumAShrInst",
    Description: "Number of AShr instructions",
  },
  {
    Index: 26,
    Name: "NumAddInst",
    Description: "Number of Add instructions",
  },
  {
    Index: 27,
    Name: "NumAllocaInst",
    Description: "Number of Alloca instructions",
  },
  {
    Index: 28,
    Name: "NumAndInst",
    Description: "Number of And instructions",
  },
  {
    Index: 29,
    Name: "BlockMid",
    Description: "Number of basic blocks with instructions between [15, 500]",
  },
  {
    Index: 30,
    Name: "BlockLow",
    Description: "Number of basic blocks with less than 15 instructions",
  },
  {
    Index: 31,
    Name: "NumBitCastInst",
    Description: "Number of BitCast instructions",
  },
  {
    Index: 32,
    Name: "NumBrInst",
    Description: "Number of Br instructions",
  },
  {
    Index: 33,
    Name: "NumCallInst",
    Description: "Number of Call instructions",
  },
  {
    Index: 34,
    Name: "NumGetElementPtrInst",
    Description: "Number of GetElementPtr instructions",
  },
  {
    Index: 35,
    Name: "NumICmpInst",
    Description: "Number of ICmp instructions",
  },
  {
    Index: 36,
    Name: "NumLShrInst",
    Description: "Number of LShr instructions",
  },
  {
    Index: 37,
    Name: "NumLoadInst",
    Description: "Number of Load instructions",
  },
  {
    Index: 38,
    Name: "NumMulInst",
    Description: "Number of Mul instructions",
  },
  {
    Index: 39,
    Name: "NumOrInst",
    Description: "Number of Or instructions",
  },
  {
    Index: 40,
    Name: "NumPHIInst",
    Description: "Number of PHI instructions",
  },
  {
    Index: 41,
    Name: "NumRetInst",
    Description: "Number of Ret instructions",
  },
  {
    Index: 42,
    Name: "NumSExtInst",
    Description: "Number of SExt instructions",
  },
  {
    Index: 43,
    Name: "NumSelectInst",
    Description: "Number of Select instructions",
  },
  {
    Index: 44,
    Name: "NumShlInst",
    Description: "Number of Shl instructions",
  },
  {
    Index: 45,
    Name: "NumStoreInst",
    Description: "Number of Store instructions",
  },
  {
    Index: 46,
    Name: "NumSubInst",
    Description: "Number of Sub instructions",
  },
  {
    Index: 47,
    Name: "NumTruncInst",
    Description: "Number of Trunc instructions",
  },
  {
    Index: 48,
    Name: "NumXorInst",
    Description: "Number of Xor instructions",
  },
  {
    Index: 49,
    Name: "NumZExtInst",
    Description: "Number of ZExt instructions",
  },
  {
    Index: 50,
    Name: "TotalBlocks",
    Description: "Number of basic blocks",
  },
  {
    Index: 51,
    Name: "TotalInsts",
    Description: "Number of instructions (of all types)",
  },
  {
    Index: 52,
    Name: "TotalMemInst",
    Description: "Number of memory instructions",
  },
  {
    Index: 53,
    Name: "TotalFuncs",
    Description: "Number of non-external functions",
  },
  {
    Index: 54,
    Name: "ArgsPhi",
    Description: "Total arguments to Phi nodes",
  },
  {
    Index: 55,
    Name: "testUnary",
    Description: "Number of Unary operations",
  },
];

export default autophaseDict;
