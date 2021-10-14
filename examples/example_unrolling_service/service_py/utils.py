# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def extract_statistics_from_ir(ir: str):
    stats = {"control_flow": 0, "arithmetic": 0, "memory": 0}
    for line in ir.splitlines():
        tokens = line.split()
        if len(tokens) > 0:
            opcode = tokens[0]
            if opcode in [
                "br",
                "call",
                "ret",
                "switch",
                "indirectbr",
                "invoke",
                "callbr",
                "resume",
                "catchswitch",
                "catchret",
                "cleanupret",
                "unreachable",
            ]:
                stats["control_flow"] += 1
            elif opcode in [
                "fneg",
                "add",
                "fadd",
                "sub",
                "fsub",
                "mul",
                "fmul",
                "udiv",
                "sdiv",
                "fdiv",
                "urem",
                "srem",
                "frem",
                "shl",
                "lshr",
                "ashr",
                "and",
                "or",
                "xor",
            ]:
                stats["arithmetic"] += 1
            elif opcode in [
                "alloca",
                "load",
                "store",
                "fence",
                "cmpxchg",
                "atomicrmw",
                "getelementptr",
            ]:
                stats["memory"] += 1

    return stats
