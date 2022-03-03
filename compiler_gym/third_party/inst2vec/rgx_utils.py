# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
# flake8: noqa
"""Helper variables and functions for regular expressions and statement tags"""
import re


########################################################################################################################
# Regex manipulation: helper functions
########################################################################################################################
def any_of(possibilities, to_add=""):
    r"""
    Helper function for regex manipulation:
    Construct a regex representing "any of" the given possibilities
    :param possibilities: list of strings representing different word possibilities
    :param to_add: string to add at the beginning of each possibility (optional)
    :return: string corresponding to regex which represents any of the given possibilities
    r"""
    assert len(possibilities) > 0
    s = r"(?:"
    if len(to_add) > 0:
        s += possibilities[0] + to_add + r" "
    else:
        s += possibilities[0]
    for i in range(len(possibilities) - 1):
        if len(to_add) > 0:
            s += r"|" + possibilities[i + 1] + to_add + r" "
        else:
            s += r"|" + possibilities[i + 1]
    return s + r")"


########################################################################################################################
# Regex manipulation: helper variables
########################################################################################################################
# Identifiers
global_id = r'(?<!%")@[r"\w\d\.\-\_\$\\]+'
local_id_no_perc = r'[r"\@\d\w\.\-\_\:]+'
local_id = r"%" + local_id_no_perc
local_or_global_id = r"(" + global_id + r"|" + local_id + r")"

# Options and linkages
linkage = any_of(
    [
        r" private",
        r" external",
        r" internal",
        r" linkonce_odr",
        r" appending",
        r" external",
        r" internal",
        r" unnamed_addr",
        r" common",
        r" hidden",
        r" weak",
        r" linkonce",
        r" extern_weak",
        r" weak_odr",
        r" private",
        r" available_externally",
        r" local_unnamed_addr",
        r" thread_local",
        r" linker_private",
    ]
)

# Immediate values
immediate_value_ad_hoc = r"#[\d\w]+"
immediate_value_true = r"true"
immediate_value_false = r"false"
immediate_value_bool = (
    r"(?:" + immediate_value_true + r"|" + immediate_value_false + r")"
)
immediate_value_int = r"(?<!\w)[-]?[0-9]+"
immediate_value_float_sci = r"(?<!\w)[-]?[0-9]+\.[0-9]+(?:e\+?-?[0-9]+)?"
immediate_value_float_hexa = r"(?<!\w)[-]?0[xX][hklmHKLM]?[A-Fa-f0-9]+"
immediate_value_float = (
    r"(?:" + immediate_value_float_sci + r"|" + immediate_value_float_hexa + r")"
)
immediate_value_vector_bool = (
    r"<i1 "
    + immediate_value_bool
    + r"(?:, i1 (?:"
    + immediate_value_bool
    + r"|undef))*>"
)
immediate_value_vector_int = (
    r"<i\d+ r"
    + immediate_value_int
    + r"(?:, i\d+ (?:"
    + immediate_value_int
    + r"|undef))*>"
)
immediate_value_vector_float = (
    r"<float "
    + immediate_value_float
    + r"(?:, float (?:"
    + immediate_value_float
    + r"|undef))*>"
)
immediate_value_vector_double = (
    r"<double "
    + immediate_value_float
    + r"(?:, double (?:"
    + immediate_value_float
    + r"|undef))*>"
)
immediate_value_string = r'(?<!\w)c".+"'
immediate_value_misc = r"(?:null|zeroinitializer)"
immediate_value = any_of(
    [
        immediate_value_true,
        immediate_value_false,
        immediate_value_int,
        immediate_value_float_sci,
        immediate_value_float_hexa,
        immediate_value_string,
        immediate_value_misc,
    ]
)
immediate_value_undef = r"undef"
immediate_value_or_undef = any_of(
    [
        immediate_value_true,
        immediate_value_false,
        immediate_value_int,
        immediate_value_float_sci,
        immediate_value_float_hexa,
        immediate_value_string,
        immediate_value_misc,
        immediate_value_ad_hoc,
        immediate_value_undef,
    ]
)

# Combos
immediate_or_local_id = any_of(
    [
        immediate_value_true,
        immediate_value_false,
        immediate_value_int,
        immediate_value_float_sci,
        immediate_value_float_hexa,
        immediate_value_vector_int,
        immediate_value_vector_float,
        immediate_value_vector_double,
        local_id,
        immediate_value_misc,
    ]
)
immediate_or_local_id_or_undef = any_of(
    [
        immediate_value_true,
        immediate_value_false,
        immediate_value_int,
        immediate_value_float_sci,
        immediate_value_float_hexa,
        immediate_value_vector_int,
        immediate_value_vector_float,
        immediate_value_vector_double,
        local_id,
        immediate_value_misc,
        immediate_value_undef,
    ]
)

# Names of aggregate types
# Lookahead so that names like '%struct.attribute_group**' won't be matched as just %struct.attribute
struct_lookahead = r"(?=[\s,\*\]\}])"
struct_name_add_on = r'(?:\([\w\d=]+\)")?'
struct_name_without_lookahead = (
    r'%[r"\@\d\w\.\-\_:]+(?:(?:<[r"\@\d\w\.\-\_:,<>\(\) \*]+>|\([r"\@\d\w\.\-\_:,<> \*]+\)|\w+)?::[r" \@\d\w\.\-\_:\)\(]*)*'
    + struct_name_add_on
)
struct_name = struct_name_without_lookahead + struct_lookahead

# Functions
func_name = r"@[\"\w\d\._\$\\]+"
func_call_pattern = r".* @[\w\d\._]+"
func_call_pattern_or_bitcast = r"(.* @[\w\d\._]+|.*bitcast .* @[\w\d\._]+ to .*)"

# new basic block
start_basic_block = (
    r"((?:<label>:)?(" + local_id_no_perc + r"):|; <label>:" + local_id_no_perc + r" )"
)

# Types
base_type = r"(?:i\d+|double|float|opaque)\**"
first_class_types = [
    r"i\d+",
    r"half",
    r"float",
    r"double",
    r"fp_128",
    r"x86_fp80",
    r"ppc_fp128",
    r"<%ID>",
]
first_class_type = any_of(first_class_types) + r"\**"
base_type_or_struct_name = any_of([base_type, struct_name_without_lookahead])
ptr_to_base_type = base_type + r"\*+"
vector_type = r"<\d+ x " + base_type + r">"
ptr_to_vector_type = vector_type + r"\*+"
array_type = r"\[\d+ x " + base_type + r"\]"
ptr_to_array_type = array_type + r"\*+"
array_of_array_type = r"\[\d+ x " + r"\[\d+ x " + base_type + r"\]" + r"\]"
struct = struct_name_without_lookahead
ptr_to_struct = struct + r"\*+"
function_type = (
    base_type
    + r" \("
    + any_of([base_type, vector_type, array_type, "..."], ",")
    + r"*"
    + any_of([base_type, vector_type, array_type, "..."])
    + r"\)\**"
)
any_type = any_of(
    [
        base_type,
        ptr_to_base_type,
        vector_type,
        ptr_to_vector_type,
        array_type,
        ptr_to_array_type,
    ]
)
any_type_or_struct = any_of(
    [
        base_type,
        ptr_to_base_type,
        vector_type,
        ptr_to_vector_type,
        array_type,
        ptr_to_array_type,
        ptr_to_struct,
    ]
)
structure_entry = any_of(
    [
        base_type,
        vector_type,
        array_type,
        array_of_array_type,
        function_type,
        r"{ .* }\**",
    ]
)
structure_entry_with_comma = any_of(
    [base_type, vector_type, array_type, array_of_array_type, function_type], ","
)
literal_structure = (
    r"(<?{ " + structure_entry_with_comma + r"*" + structure_entry + r" }>?|{})"
)

# Tokens
unknown_token = r"!UNK"  # starts with '!' to guarantee it will appear first in the alphabetically sorted vocabulary

########################################################################################################################
# Tags for clustering statements (by statement semantics) and helper functions
########################################################################################################################
# List of families of operations
llvm_IR_stmt_families = [
    # [r"tag level 1",                  r"tag level 2",            r"tag level 3",              r"regex"                    ]
    [r"unknown token", "unknown token", "unknown token", "!UNK"],
    [r"integer arithmetic", "addition", "add integers", "<%ID> = add .*"],
    [r"integer arithmetic", "subtraction", "subtract integers", "<%ID> = sub .*"],
    [
        r"integer arithmetic",
        r"multiplication",
        r"multiply integers",
        r"<%ID> = mul .*",
    ],
    [
        r"integer arithmetic",
        r"division",
        r"unsigned integer division",
        r"<%ID> = udiv .*",
    ],
    [
        r"integer arithmetic",
        r"division",
        r"signed integer division",
        r"<%ID> = sdiv .*",
    ],
    [
        r"integer arithmetic",
        r"remainder",
        r"remainder of signed div",
        r"<%ID> = srem .*",
    ],
    [
        r"integer arithmetic",
        r"remainder",
        r"remainder of unsigned div.",
        r"<%ID> = urem .*",
    ],
    [r"floating-point arithmetic", "addition", "add floats", "<%ID> = fadd .*"],
    [
        r"floating-point arithmetic",
        r"subtraction",
        r"subtract floats",
        r"<%ID> = fsub .*",
    ],
    [
        r"floating-point arithmetic",
        r"multiplication",
        r"multiply floats",
        r"<%ID> = fmul .*",
    ],
    [r"floating-point arithmetic", "division", "divide floats", "<%ID> = fdiv .*"],
    [r"bitwise arithmetic", "and", "and", "<%ID> = and .*"],
    [r"bitwise arithmetic", "or", "or", "<%ID> = or .*"],
    [r"bitwise arithmetic", "xor", "xor", "<%ID> = xor .*"],
    [r"bitwise arithmetic", "shift left", "shift left", "<%ID> = shl .*"],
    [r"bitwise arithmetic", "arithmetic shift right", "ashr", "<%ID> = ashr .*"],
    [
        r"bitwise arithmetic",
        r"logical shift right",
        r"logical shift right",
        r"<%ID> = lshr .*",
    ],
    [
        r"comparison operation",
        r"compare integers",
        r"compare integers",
        r"<%ID> = icmp .*",
    ],
    [
        r"comparison operation",
        r"compare floats",
        r"compare floats",
        r"<%ID> = fcmp .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast single val",
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque) .* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast single val*",
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\* .* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast single val**",
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\* .* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast single val***",
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\* .* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast single val****",
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\*\* .* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast array",
        r"<%ID> = bitcast \[\d.* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast vector",
        r"<%ID> = bitcast <\d.* to .*",
    ],
    [
        r"conversion operation",
        r"bitcast",
        r"bitcast structure",
        r'<%ID> = bitcast (%"|<{|<%|{).* to .*',
    ],
    [r"conversion operation", "bitcast", "bitcast void", "<%ID> = bitcast void "],
    [
        r"conversion operation",
        r"extension/truncation",
        r"extend float",
        r"<%ID> = fpext .*",
    ],
    [
        r"conversion operation",
        r"extension/truncation",
        r"truncate floats",
        r"<%ID> = fptrunc .*",
    ],
    [
        r"conversion operation",
        r"extension/truncation",
        r"sign extend ints",
        r"<%ID> = sext .*",
    ],
    [
        r"conversion operation",
        r"extension/truncation",
        r"truncate int to ... ",
        r"<%ID> = trunc .* to .*",
    ],
    [
        r"conversion operation",
        r"extension/truncation",
        r"zero extend integers",
        r"<%ID> = zext .*",
    ],
    [
        r"conversion operation",
        r"convert",
        r"convert signed integers to... ",
        r"<%ID> = sitofp .*",
    ],
    [
        r"conversion operation",
        r"convert",
        r"convert unsigned integer to... ",
        r"<%ID> = uitofp .*",
    ],
    [
        r"conversion operation",
        r"convert int to ptr",
        r"convert int to ptr",
        r"<%ID> = inttoptr .*",
    ],
    [
        r"conversion operation",
        r"convert ptr to int",
        r"convert ptr to int",
        r"<%ID> = ptrtoint .*",
    ],
    [
        r"conversion operation",
        r"convert floats",
        r"convert float to sint",
        r"<%ID> = fptosi .*",
    ],
    [
        r"conversion operation",
        r"convert floats",
        r"convert float to uint",
        r"<%ID> = fptoui .*",
    ],
    [r"control flow", "phi", "phi", "<%ID> = phi .*"],
    [
        r"control flow",
        r"switch",
        r"jump table line",
        r"i\d{1,2} <(INT|FLOAT)>, label <%ID>",
    ],
    [r"control flow", "select", "select", "<%ID> = select .*"],
    [r"control flow", "invoke", "invoke and ret type", "<%ID> = invoke .*"],
    [r"control flow", "invoke", "invoke void", "invoke (fastcc )?void .*"],
    [r"control flow", "branch", "branch conditional", "br i1 .*"],
    [r"control flow", "branch", "branch unconditional", "br label .*"],
    [r"control flow", "branch", "branch indirect", "indirectbr .*"],
    [r"control flow", "control flow", "switch", "switch .*"],
    [r"control flow", "return", "return", "ret .*"],
    [r"control flow", "resume", "resume", "resume .*"],
    [r"control flow", "unreachable", "unreachable", "unreachable.*"],
    [r"control flow", "exception handling", "catch block", "catch .*"],
    [r"control flow", "exception handling", "cleanup clause", "cleanup"],
    [
        r"control flow",
        r"exception handling",
        r"landingpad for exceptions",
        r"<%ID> = landingpad .",
    ],
    [
        r"function",
        r"function call",
        r"sqrt (llvm-intrinsic)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>) @(llvm|llvm\..*)\.sqrt.*",
    ],
    [
        r"function",
        r"function call",
        r"fabs (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fabs.*",
    ],
    [
        r"function",
        r"function call",
        r"max (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.max.*",
    ],
    [
        r"function",
        r"function call",
        r"min (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.min.*",
    ],
    [
        r"function",
        r"function call",
        r"fma (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fma.*",
    ],
    [
        r"function",
        r"function call",
        r"phadd (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.phadd.*",
    ],
    [
        r"function",
        r"function call",
        r"pabs (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pabs.*",
    ],
    [
        r"function",
        r"function call",
        r"pmulu (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pmulu.*",
    ],
    [
        r"function",
        r"function call",
        r"umul (llvm-intr.)",
        r"<%ID> = (tail |musttail |notail )?call {.*} @llvm\.umul.*",
    ],
    [
        r"function",
        r"function call",
        r"prefetch (llvm-intr.)",
        r"(tail |musttail |notail )?call void @llvm\.prefetch.*",
    ],
    [
        r"function",
        r"function call",
        r"trap (llvm-intr.)",
        r"(tail |musttail |notail )?call void @llvm\.trap.*",
    ],
    [r"function", "func decl / def", "function declaration", "declare .*"],
    [r"function", "func decl / def", "function definition", "define .*"],
    [
        r"function",
        r"function call",
        r"function call void",
        r"(tail |musttail |notail )?call( \w+)? void [\w\)\(\}\{\.\,\*\d\[\]\s<>%]*(<[@%]ID>\(|.*bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call mem lifetime",
        r"(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.lifetime.*",
    ],
    [
        r"function",
        r"function call",
        r"function call mem copy",
        r"(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memcpy\..*",
    ],
    [
        r"function",
        r"function call",
        r"function call mem set",
        r"(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memset\..*",
    ],
    [
        r"function",
        r"function call",
        r"function call single val",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80|<\d+ x (i\d+|float|double)>) (.*<[@%]ID>\(|(\(.*\) )?bitcast ).*",
    ],
    [
        r"function",
        r"function call",
        r"function call single val*",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\* (.*<[@%]ID>\(|\(.*\) bitcast ).*",
    ],
    [
        r"function",
        r"function call",
        r"function call single val**",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\*\* (.*<[@%]ID>\(|\(.*\) bitcast ).*",
    ],
    [
        r"function",
        r"function call",
        r"function call array",
        r"<%ID> = (tail |musttail |notail )?call[^{]* \[.*\] (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call array*",
        r"<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call array**",
        r"<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\*\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call structure",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>) (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call structure*",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call structure**",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call structure***",
        r"<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        r"function",
        r"function call",
        r"function call asm value",
        r"<%ID> = (tail |musttail |notail )?call.* asm .*",
    ],
    [
        r"function",
        r"function call",
        r"function call asm void",
        r"(tail |musttail |notail )?call void asm .*",
    ],
    [
        r"function",
        r"function call",
        r"function call function",
        r"<%ID> = (tail |musttail |notail )?call[^{]* void \([^\(\)]*\)\** <[@%]ID>\(",
    ],
    [
        r"global variables",
        r"glob. var. definition",
        r"???",
        r"<@ID> = (?!.*constant)(?!.*alias).*",
    ],
    [r"global variables", "constant definition", "???", "<@ID> = .*constant .*"],
    [
        r"memory access",
        r"load from memory",
        r"load structure",
        r'<%ID> = load (\w* )?(%"|<\{|\{ <|\{ \[|\{ |<%|opaque).*',
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val*",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val**",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val***",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val****",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val*****",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val******",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load single val*******",
        r"<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*\*[, ].*",
    ],
    [
        r"memory access",
        r"load from memory",
        r"load vector",
        r"<%ID> = load <\d+ x .*",
    ],
    ["memory access", "load from memory", "load array", r"<%ID> = load \[\d.*"],
    [
        r"memory access",
        r"load from memory",
        r"load fction ptr",
        r"<%ID> = load void \(",
    ],
    [r"memory access", "store", "store", "store.*"],
    [r"memory addressing", "GEP", "GEP", r"<%ID> = getelementptr .*"],
    [
        r"memory allocation",
        r"allocate on stack",
        r"allocate structure",
        r'<%ID> = alloca (%"|<{|<%|{ |opaque).*',
    ],
    [
        r"memory allocation",
        r"allocate on stack",
        r"allocate vector",
        r"<%ID> = alloca <\d.*",
    ],
    [
        r"memory allocation",
        r"allocate on stack",
        r"allocate array",
        r"<%ID> = alloca \[\d.*",
    ],
    [
        r"memory allocation",
        r"allocate on stack",
        r"allocate single value",
        r"<%ID> = alloca (double|float|i\d{1,3})\*?.*",
    ],
    [
        r"memory allocation",
        r"allocate on stack",
        r"allocate void",
        r"<%ID> = alloca void \(.*",
    ],
    [
        r"memory atomics",
        r"atomic memory modify",
        r"atomicrw xchg",
        r"<%ID> = atomicrmw.* xchg .*",
    ],
    [
        r"memory atomics",
        r"atomic memory modify",
        r"atomicrw add",
        r"<%ID> = atomicrmw.* add .*",
    ],
    [
        r"memory atomics",
        r"atomic memory modify",
        r"atomicrw sub",
        r"<%ID> = atomicrmw.* sub .*",
    ],
    [
        r"memory atomics",
        r"atomic memory modify",
        r"atomicrw or",
        r"<%ID> = atomicrmw.* or .*",
    ],
    [
        r"memory atomics",
        r"atomic compare exchange",
        r"cmpxchg single val",
        r"<%ID> = cmpxchg (weak )?(i\d+|float|double|x86_fp80)\*",
    ],
    [
        r"non-instruction",
        r"label",
        r"label declaration",
        r"; <label>:.*(\s+; preds = <LABEL>)?",
    ],
    [
        r"non-instruction",
        r"label",
        r"label declaration",
        r"<LABEL>:( ; preds = <LABEL>)?",
    ],
    [
        r"value aggregation",
        r"extract value",
        r"extract value",
        r"<%ID> = extractvalue .*",
    ],
    [
        r"value aggregation",
        r"insert value",
        r"insert value",
        r"<%ID> = insertvalue .*",
    ],
    [
        r"vector operation",
        r"insert element",
        r"insert element",
        r"<%ID> = insertelement .*",
    ],
    [
        r"vector operation",
        r"extract element",
        r"extract element",
        r"<%ID> = extractelement .*",
    ],
    [
        r"vector operation",
        r"shuffle vector",
        r"shuffle vector",
        r"<%ID> = shufflevector .*",
    ],
]


# Helper functions for exploring llvm_IR_families
def get_list_tag_level_1():
    r"""
    Get the list of all level-1 tags in the data structure llvm_IR_families

    :return: list containing strings corresponding to all level 1 tags
    r"""
    list_tags = list()
    for fam in llvm_IR_stmt_families:
        list_tags.append(fam[0])

    return list(set(list_tags))


def get_list_tag_level_2(tag_level_1="all"):
    r"""
    Get the list of all level-2 tags in the data structure llvm_IR_families
    corresponding to the string given as an input, or absolutely all of them
    if input == r'all'

    :param tag_level_1: string containing the level-1 tag to query, or 'all'
    :return: list of strings
    r"""

    # Make sure the input parameter is valid
    assert tag_level_1 in get_list_tag_level_1() or tag_level_1 == r"all", (
        tag_level_1 + r" invalid"
    )

    list_tags = list()

    if tag_level_1 == r"all":
        for fam in llvm_IR_stmt_families:
            list_tags.append(fam[1])
        list_tags = sorted(set(list_tags))
    else:
        for fam in llvm_IR_stmt_families:
            if fam[0] == tag_level_1:
                list_tags.append(fam[1])

    return list(set(list_tags))


########################################################################################################################
# Tags for clustering statements (by statement type)
########################################################################################################################
# Helper lists
types_int = [r"i1", "i8", "i16", "i32", "i64"]
types_flpt = [r"half", "float", "double", "fp128", "x86_fp80", "ppc_fp128"]
fast_math_flag = [
    r"",
    r"nnan ",
    r"ninf ",
    r"nsz ",
    r"arcp ",
    r"contract ",
    r"afn ",
    r"reassoc ",
    r"fast ",
]
opt_load = [r"atomic ", "volatile "]
opt_addsubmul = [r"nsw ", "nuw ", "nuw nsw "]
opt_usdiv = [r"", "exact "]
opt_icmp = [
    r"eq ",
    r"ne ",
    r"ugt ",
    r"uge ",
    r"ult ",
    r"ule ",
    r"sgt ",
    r"sge ",
    r"slt ",
    r"sle ",
]
opt_fcmp = [
    r"false ",
    r"oeq ",
    r"ogt ",
    r"oge ",
    r"olt ",
    r"olt ",
    r"ole ",
    r"one ",
    r"ord ",
    r"ueq ",
    r"ugt ",
    r"uge ",
    r"ult ",
    r"ule ",
    r"une ",
    r"uno ",
    r"true ",
]
opt_define = [
    r"",
    r"linkonce_odr ",
    r"linkonce_odr ",
    r"zeroext ",
    r"dereferenceable\(\d+\) ",
    r"hidden ",
    r"internal ",
    r"nonnull ",
    r"weak_odr ",
    r"fastcc ",
    r"noalias ",
    r"signext ",
    r"spir_kernel ",
]
opt_invoke = [
    r"",
    r"dereferenceable\(\d+\) ",
    r"noalias ",
    r"fast ",
    r"zeroext ",
    r"signext ",
    r"fastcc ",
]
opt_GEP = [r"", "inbounds "]


# Helper functions
def any_of(possibilities, to_add=""):
    r"""
    Construct a regex representing "any of" the given possibilities
    :param possibilities: list of strings representing different word possibilities
    :param to_add: string to add at the beginning of each possibility (optional)
    :return: string corresponding to regex which represents any of the given possibilities
    r"""
    assert len(possibilities) > 0
    s = r"("
    if len(to_add) > 0:
        s += possibilities[0] + to_add + r" "
    else:
        s += possibilities[0]
    for i in range(len(possibilities) - 1):
        if len(to_add) > 0:
            s += r"|" + possibilities[i + 1] + to_add + r" "
        else:
            s += r"|" + possibilities[i + 1]
    return s + r")"


# Main tags
llvm_IR_stmt_tags = [
    # ['regex'                                                    r'tag'                   r'tag general'
    [
        r"<@ID> = (?!.*constant)(?!.*alias).*",
        r"global definition",
        r"global variable definition",
    ],
    [r"<@ID> = .*constant .*", "global const. def.", "global variable definition"],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = add " + any_of(opt_addsubmul) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sub " + any_of(opt_addsubmul) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = mul " + any_of(opt_addsubmul) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i16> .*",
        r"<d x i16>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i32> .*",
        r"<d x i32>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i64> .*",
        r"<d x i64>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = udiv " + any_of(opt_usdiv) + r"?<\d+ x i128> .*",
        r"<d x i128>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i16> .*",
        r"<d x i16>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i32> .*",
        r"<d x i32>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i64> .*",
        r"<d x i64>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = sdiv " + any_of(opt_usdiv) + r"?<\d+ x i128> .*",
        r"<d x i128>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<%ID> .*",
        r"struct  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<%ID>\* .*",
        r"struct*  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<%ID>\*\* .*",
        r"struct**  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<%ID>\*\*\* .*",
        r"struct***  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i24 .*",
        r"i24 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i24> .*",
        r"<d x i24> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i40 .*",
        r"i40 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i40> .*",
        r"<d x i40> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i1\* .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i2\* .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i4\* .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i8\* .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i16\* .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i32\* .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i40\* .*",
        r"i40* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i64\* .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i128\* .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?x86_fp80\* .*",
        r"float* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?float\* .*",
        r"float* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i1\*\* .*",
        r"i1**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i2\*\* .*",
        r"i2**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i4\*\* .*",
        r"i4**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i8\*\* .*",
        r"i8**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i16\*\* .*",
        r"i16** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i32\*\* .*",
        r"i32** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i40\*\* .*",
        r"i40** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i64\*\* .*",
        r"i64** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?i128\*\* .*",
        r"i128** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?x86_fp80\*\* .*",
        r"float** operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?float\*\* .*",
        r"float** operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?double\*\* .*",
        r"double** operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<%ID>\* .*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r'?(%"|opaque).*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = icmp " + any_of(opt_icmp) + r"?void \(.*",
        r"function op",
        r"struct/class op",
    ],
    [r"<%ID> = srem i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = srem <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    [r"<%ID> = srem i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = srem <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    [r"<%ID> = srem i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = srem <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    [r"<%ID> = srem i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = srem <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    [r"<%ID> = srem i16 .*", "i16 operation", "int operation"],
    [
        r"<%ID> = srem <\d+ x i16> .*",
        r"<d x i16>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = srem i32 .*", "i32 operation", "int operation"],
    [
        r"<%ID> = srem <\d+ x i32> .*",
        r"<d x i32>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = srem i64 .*", "i64 operation", "int operation"],
    [
        r"<%ID> = srem <\d+ x i64> .*",
        r"<d x i64>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = srem i128 .*", "i128 operation", "int operation"],
    [
        r"<%ID> = srem <\d+ x i128> .*",
        r"<d x i128>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = urem i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = urem <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    [r"<%ID> = urem i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = urem <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    [r"<%ID> = urem i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = urem <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    [r"<%ID> = urem i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = urem <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    [r"<%ID> = urem i16 .*", "i16 operation", "int operation"],
    [
        r"<%ID> = urem <\d+ x i16> .*",
        r"<d x i16>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = urem i32 .*", "i32 operation", "int operation"],
    [
        r"<%ID> = urem <\d+ x i32> .*",
        r"<d x i32>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = urem i64 .*", "i32 operation", "int operation"],
    [
        r"<%ID> = urem <\d+ x i64> .*",
        r"<d x i64>  operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = urem i128 .*", "i128 operation", "int operation"],
    [
        r"<%ID> = urem <\d+ x i128> .*",
        r"<d x i128>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fadd " + any_of(fast_math_flag) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fsub " + any_of(fast_math_flag) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fmul " + any_of(fast_math_flag) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fdiv " + any_of(fast_math_flag) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = frem " + any_of(fast_math_flag) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?x86_fp80.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?<\d+ x x86_fp80>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?float.*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?<\d+ x float>.*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?double.*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + r"?<\d+ x double>.*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [r"<%ID> = atomicrmw add i1\* .*", "i1* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i2\* .*", "i2* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i4\* .*", "i4* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i8\* .*", "i8* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i16\* .*", "i16* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i32\* .*", "i32* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i64\* .*", "i64* operation", "int* operation"],
    [r"<%ID> = atomicrmw add i128\* .*", "i128* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i1\* .*", "i1* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i2\* .*", "i2* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i4\* .*", "i4* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i8\* .*", "i8* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i16\* .*", "i16* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i32\* .*", "i32* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i64\* .*", "i64* operation", "int* operation"],
    [r"<%ID> = atomicrmw sub i128\* .*", "i128* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i1\* .*", "i1* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i2\* .*", "i2* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i4\* .*", "i4* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i8\* .*", "i8* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i16\* .*", "i16* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i32\* .*", "i32* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i64\* .*", "i64* operation", "int* operation"],
    [r"<%ID> = atomicrmw or i128\* .*", "i128* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i1\* .*", "i1* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i2\* .*", "i2* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i4\* .*", "i4* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i8\* .*", "i8* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i16\* .*", "i16* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i32\* .*", "i32* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i64\* .*", "i64* operation", "int* operation"],
    [r"<%ID> = atomicrmw xchg i128\* .*", "i128* operation", "int* operation"],
    [r"<%ID> = alloca i1($|,).*", "i1  operation", "int operation"],
    [r"<%ID> = alloca i2($|,).*", "i2  operation", "int operation"],
    [r"<%ID> = alloca i4($|,).*", "i4  operation", "int operation"],
    [r"<%ID> = alloca i8($|,).*", "i8  operation", "int operation"],
    [r"<%ID> = alloca i16($|,).*", "i16 operation", "int operation"],
    [r"<%ID> = alloca i32($|,).*", "i32 operation", "int operation"],
    [r"<%ID> = alloca i64($|,).*", "i64 operation", "int operation"],
    [r"<%ID> = alloca i128($|,).*", "i128 operation", "int operation"],
    [r"<%ID> = alloca i1\*($|,).*", "i1*  operation", "int* operation"],
    [r"<%ID> = alloca i2\*($|,).*", "i2*  operation", "int* operation"],
    [r"<%ID> = alloca i4\*($|,).*", "i4*  operation", "int* operation"],
    [r"<%ID> = alloca i8\*($|,).*", "i8*  operation", "int* operation"],
    [r"<%ID> = alloca i16\*($|,).*", "i16* operation", "int* operation"],
    [r"<%ID> = alloca i32\*($|,).*", "i32* operation", "int* operation"],
    [r"<%ID> = alloca i64\*($|,).*", "i64* operation", "int* operation"],
    [r"<%ID> = alloca i128\*($|,).*", "i128* operation", "int* operation"],
    [
        r"<%ID> = alloca x86_fp80($|,).*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = alloca float($|,).*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = alloca double($|,).*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = alloca x86_fp80\*($|,).*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = alloca float\*($|,).*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = alloca double\*($|,).*",
        r"double* operation",
        r"floating point* operation",
    ],
    ['<%ID> = alloca %".*', "struct/class op", "struct/class op"],
    [r"<%ID> = alloca <%.*", "struct/class op", "struct/class op"],
    [r"<%ID> = alloca <?{.*", "struct/class op", "struct/class op"],
    [r"<%ID> = alloca opaque.*", "struct/class op", "struct/class op"],
    [
        r"<%ID> = alloca <\d+ x i1>, .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i2>, .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i4>, .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i8>, .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i16>, .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i32>, .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i64>, .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i128>, .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x x86_fp80>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x float>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x double>, .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x \{ .* \}>, .*",
        r"<d x structure> operation",
        r"<d x structure> operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i1>\*, .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i2>\*, .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i4>\*, .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i8>\*, .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i16>\*, .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i32>\*, .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i64>\*, .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x i128>\*, .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x x86_fp80>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x float>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x double>\*, .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = alloca <\d+ x \{ .* \}>\*, .*",
        r"<d x structure>* operation",
        r"<d x structure>* operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i1\], .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i2\], .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i4\], .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i8\], .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i16\], .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i32\], .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i64\], .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x i128\], .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x x86_fp80\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x float\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x double\], .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = alloca \[\d+ x \{ .* \}\], .*",
        r"[d x structure] operation",
        r"[d x structure] operation",
    ],
    [
        r"<%ID> = alloca { { float, float } }, .*",
        r"{ float, float } operation",
        r"complex operation",
    ],
    [
        r"<%ID> = alloca { { double, double } }, .*",
        r"{ double, double } operation",
        r"complex operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i1, .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i2, .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i4, .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i8, .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i16, .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i24, .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i32, .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i40, .*",
        r"i40 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i64, .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i128, .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i256, .*",
        r"i256 operation",
        r"int operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i1\*, .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i2\*, .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i4\*, .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i8\*, .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i16\*, .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i24\*, .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i32\*, .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i40\*, .*",
        r"i40* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i64\*, .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i128\*, .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i256\*, .*",
        r"i256* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i1\*\*, .*",
        r"i1**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i2\*\*, .*",
        r"i2**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i4\*\*, .*",
        r"i4**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i8\*\*, .*",
        r"i8**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i16\*\*, .*",
        r"i16** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i24\*\*, .*",
        r"i16** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i32\*\*, .*",
        r"i32** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i40\*\*, .*",
        r"i40** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i64\*\*, .*",
        r"i64** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i128\*\*, .*",
        r"i128** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i256\*\*, .*",
        r"i256** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i1\*\*\*, .*",
        r"i1***  operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i2\*\*\*, .*",
        r"i2***  operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i4\*\*\*, .*",
        r"i4***  operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i8\*\*\*, .*",
        r"i8***  operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i16\*\*\*, .*",
        r"i16*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i24\*\*\*, .*",
        r"i16*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i32\*\*\*, .*",
        r"i32*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i40\*\*\*, .*",
        r"i40*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i64\*\*\*, .*",
        r"i64*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i128\*\*\*, .*",
        r"i128*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?i256\*\*\*, .*",
        r"i256*** operation",
        r"int*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?x86_fp80, .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?float, .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?double, .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?x86_fp80\*, .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?float\*, .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?double\*, .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?x86_fp80\*\*, .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?float\*\*, .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?double\*\*, .*",
        r"double** operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?x86_fp80\*\*\*, .*",
        r"float***  operation",
        r"floating point*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?float\*\*\*, .*",
        r"float***  operation",
        r"floating point*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?double\*\*\*, .*",
        r"double*** operation",
        r"floating point*** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r'?%".*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<%.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i1>, .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i2>, .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i4>, .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i8>, .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i16>, .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i24>, .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i32>, .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i40>, .*",
        r"<d x i40> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i64>, .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i128>, .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x x86_fp80>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x float>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x double>, .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x \{ .* \}>, .*",
        r"<d x structure> operation",
        r"<d x structure> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i1\*>, .*",
        r"<d x i1*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i2\*>, .*",
        r"<d x i2*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i4\*>, .*",
        r"<d x i4*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i8\*>, .*",
        r"<d x i8*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i16\*>, .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i24\*>, .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i32\*>, .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i40\*>, .*",
        r"<d x i40*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i64\*>, .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i128\*>, .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x x86_fp80\*>, .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x float\*>, .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x double\*>, .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i1>\*, .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i2>\*, .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i4>\*, .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i8>\*, .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i16>\*, .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i24>\*, .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i32>\*, .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i40>\*, .*",
        r"<d x i40>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i64>\*, .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x i128>\*, .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x x86_fp80>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x float>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x double>\*, .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x \{ .* \}>\*, .*",
        r"<d x structure>* operation",
        r"<d x structure>* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x x86_fp80>\*\*, .*",
        r"<d x float>** operation",
        r"<d x floating point>** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x float>\*\*, .*",
        r"<d x float>** operation",
        r"<d x floating point>** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x double>\*\*, .*",
        r"<d x double>** operation",
        r"<d x floating point>** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?<\d+ x \{ .* \}>\*\*, .*",
        r"<d x structure>** operation",
        r"<d x structure>** operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i1\], .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i2\], .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i4\], .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i8\], .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i16\], .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i24\], .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i32\], .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i40\], .*",
        r"[d x i40] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i64\], .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i128\], .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x x86_fp80\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x float\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x double\], .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x \{ .* \}\], .*",
        r"[d x structure] operation",
        r"[d x structure] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i1\]\*, .*",
        r"[d x i1]*  operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i2\]\*, .*",
        r"[d x i2]*  operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i4\]\*, .*",
        r"[d x i4]*  operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i8\]\*, .*",
        r"[d x i8]*  operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i16\]\*, .*",
        r"[d x i16]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i32\]\*, .*",
        r"[d x i32]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i40\]\*, .*",
        r"[d x i40]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i64\]\*, .*",
        r"[d x i64]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x i128\]\*, .*",
        r"[d x i128]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x x86_fp80\]\*, .*",
        r"[d x float]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x float\]\*, .*",
        r"[d x float]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x double\]\*, .*",
        r"[d x double]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?\[\d+ x \{ .* \}\]\*, .*",
        r"[d x structure]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = load " + any_of(opt_load) + r"?.*\(.*\)\*+, .*",
        r"function operation",
        r"function operation",
    ],
    [r"store " + any_of(opt_load) + r"?i1 .*", "i1  operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i2 .*", "i2  operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i4 .*", "i4  operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i8 .*", "i8  operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i16 .*", "i16 operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i24 .*", "i16 operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i32 .*", "i32 operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i40 .*", "i32 operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i64 .*", "i64 operation", "int operation"],
    [r"store " + any_of(opt_load) + r"?i128 .*", "i128 operation", "int operation"],
    [
        r"store " + any_of(opt_load) + r"?i1\* .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i2\* .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i4\* .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i8\* .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i16\* .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i32\* .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i64\* .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i128\* .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i1\*\* .*",
        r"i1**  operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i2\*\* .*",
        r"i2**  operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i4\*\* .*",
        r"i4**  operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i8\*\* .*",
        r"i8**  operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i16\*\* .*",
        r"i16** operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i32\*\* .*",
        r"i32** operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i64\*\* .*",
        r"i64** operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?i128\*\* .*",
        r"i128** operation",
        r"int** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?x86_fp80 .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?float .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?double .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?float\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?x86_fp80\*\* .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?float\*\* .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?double\*\* .*",
        r"double** operation",
        r"floating point** operation",
    ],
    [r"store " + any_of(opt_load) + r"?void \(.*", "function op", "function op"],
    [r"store " + any_of(opt_load) + r'?%".*', "struct/class op", "struct/class op"],
    [r"store " + any_of(opt_load) + r"?<%.*", "struct/class op", "struct/class op"],
    [
        r"store " + any_of(opt_load) + r"?<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"store " + any_of(opt_load) + r"?opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x \{ .* \}> .*",
        r"<d x \{ .* \}> operation",
        r"<d x \{ .* \}> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i1\*> .*",
        r"<d x i1*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i2\*> .*",
        r"<d x i2*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i4\*> .*",
        r"<d x i4*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i8\*> .*",
        r"<d x i8*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i16\*> .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i32\*> .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i64\*> .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i128\*> .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x x86_fp80\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x float\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x double\*> .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x \{ .* \}\*> .*",
        r"<d x \{ .* \}*> operation",
        r"<d x \{ .* \}*> operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i1>\* .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i2>\* .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i4>\* .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i8>\* .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i16>\* .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i32>\* .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i64>\* .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x i128>\* .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x x86_fp80>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x \{ .* \}\*?>\* .*",
        r"<d x struct>* operation",
        r"<d x \{ .* \}>* operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?<\d+ x void \(.*",
        r"<d x function>* operation",
        r"<d x function operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i1\] .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i2\] .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i4\] .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i8\] .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i16\] .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i32\] .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i64\] .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x i128\] .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x x86_fp80\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x float\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x double\] .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"store " + any_of(opt_load) + r"?\[\d+ x \{ .* \}\] .*",
        r"[d x structure] operation",
        r"[d x structure] operation",
    ],
    [r"declare (noalias |nonnull )*void .*", "void operation", "void operation"],
    [r"declare (noalias |nonnull )*i1 .*", "i1  operation", "int operation"],
    [r"declare (noalias |nonnull )*i2 .*", "i2  operation", "int operation"],
    [r"declare (noalias |nonnull )*i4 .*", "i4  operation", "int operation"],
    [r"declare (noalias |nonnull )*i8 .*", "i8  operation", "int operation"],
    [r"declare (noalias |nonnull )*i16 .*", "i16 operation", "int operation"],
    [r"declare (noalias |nonnull )*i32 .*", "i32 operation", "int operation"],
    [r"declare (noalias |nonnull )*i64 .*", "i64 operation", "int operation"],
    [r"declare (noalias |nonnull )*i8\* .*", "i8*  operation", "int* operation"],
    [r"declare (noalias |nonnull )*i16\* .*", "i16* operation", "int* operation"],
    [r"declare (noalias |nonnull )*i32\* .*", "i32* operation", "int* operation"],
    [r"declare (noalias |nonnull )*i64\* .*", "i64* operation", "int* operation"],
    [
        r"declare (noalias |nonnull )*x86_fp80 .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"declare (noalias |nonnull )*float .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"declare (noalias |nonnull )*double .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"declare (noalias |nonnull )*x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"declare (noalias |nonnull )*float\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"declare (noalias |nonnull )*double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    ['declare (noalias |nonnull )*%".*', "struct/class op", "struct/class op"],
    [r"declare (noalias |nonnull )*<%.*", "struct/class op", "struct/class op"],
    [r"declare (noalias |nonnull )*<?{.*", "struct/class op", "struct/class op"],
    [
        r"declare (noalias |nonnull )*opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i1>\* .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i2>\* .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i4>\* .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i8>\* .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i16>\* .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i32>\* .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i64>\* .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x i128>\* .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x x86_fp80>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"declare (noalias |nonnull )*<\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i1\] .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i2\] .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i4\] .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i8\] .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i16\] .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i32\] .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i64\] .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x i128\] .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x x86_fp80\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x float\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"declare (noalias |nonnull )*\[\d+ x double\] .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+void .*",
        r"void operation",
        r"void operation",
    ],
    [r"define " + any_of(opt_define) + r"+i1 .*", "i1  operation", "int operation"],
    [r"define " + any_of(opt_define) + r"+i2 .*", "i2  operation", "int operation"],
    [r"define " + any_of(opt_define) + r"+i4 .*", "i4  operation", "int operation"],
    [r"define " + any_of(opt_define) + r"+i8 .*", "i8  operation", "int operation"],
    [
        r"define " + any_of(opt_define) + r"+i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i1\* .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i2\* .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i4\* .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i8\* .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i16\* .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i32\* .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i64\* .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+i128\* .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+x86_fp80 .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+float .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+double .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+float\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"define " + any_of(opt_define) + r'+%".*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"define " + any_of(opt_define) + r"+<%.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"define " + any_of(opt_define) + r"+<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"define " + any_of(opt_define) + r"+opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i1>\* .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i2>\* .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i4>\* .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i8>\* .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i16>\* .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i32>\* .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i64>\* .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x i128>\* .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x x86_fp80>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+<\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i1\] .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i2\] .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i4\] .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i8\] .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i16\] .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i32\] .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i64\] .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x i128\] .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x x86_fp80\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x float\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"define " + any_of(opt_define) + r"+\[\d+ x double\] .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i1\* .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i2\* .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i4\* .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i8\* .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i16\* .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i32\* .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i64\* .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i128\* .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i1\*\* .*",
        r"i1**  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i2\*\* .*",
        r"i2**  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i4\*\* .*",
        r"i4**  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*i8\*\* .*",
        r"i8**  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*i16\*\* .*",
        r"i16** operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*i32\*\* .*",
        r"i32** operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*i64\*\* .*",
        r"i64** operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*i128\*\* .*",
        r"i128** operation",
        r"int* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*x86_fp80 .*",
        r"float operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*float .*",
        r"float operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*double .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*x86_fp80\* .*",
        r"float* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*float\* .*",
        r"float* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*x86_fp80\*\* .*",
        r"float** operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*float\*\* .*",
        r"float** operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*double\*\* .*",
        r"double** operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r'*%".*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*<%.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + r"*opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i1>\* .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i2>\* .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i4>\* .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i8>\* .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i16>\* .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i32>\* .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i64>\* .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x i128>\* .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x x86_fp80>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*<\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i1\] .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i2\] .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i4\] .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i8\] .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i16\] .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i32\] .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i64\] .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x i128\] .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x x86_fp80\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x float\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + r"*\[\d+ x double\] .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [r"ret i1 .*", "i1  operation", "int operation"],
    [r"ret i2 .*", "i2  operation", "int operation"],
    [r"ret i4 .*", "i4  operation", "int operation"],
    [r"ret i8 .*", "i8  operation", "int operation"],
    [r"ret i16 .*", "i16 operation", "int operation"],
    [r"ret i32 .*", "i32 operation", "int operation"],
    [r"ret i64 .*", "i64 operation", "int operation"],
    [r"ret i128 .*", "i128 operation", "int operation"],
    [r"ret i1\* .*", "i1*  operation", "int* operation"],
    [r"ret i2\* .*", "i2*  operation", "int* operation"],
    [r"ret i4\* .*", "i4*  operation", "int* operation"],
    [r"ret i8\* .*", "i8*  operation", "int* operation"],
    [r"ret i16\* .*", "i16* operation", "int* operation"],
    [r"ret i32\* .*", "i32* operation", "int* operation"],
    [r"ret i64\* .*", "i64* operation", "int* operation"],
    [r"ret i128\* .*", "i128* operation", "int* operation"],
    [r"ret x86_fp80 .*", "x86_fp80  operation", "floating point operation"],
    [r"ret float .*", "float  operation", "floating point operation"],
    [r"ret double .*", "double operation", "floating point operation"],
    [r"ret x86_fp80\* .*", "x86_fp80*  operation", "floating point* operation"],
    [r"ret float\* .*", "float*  operation", "floating point* operation"],
    [r"ret double\* .*", "double* operation", "floating point* operation"],
    ['ret %".*', "struct/class op", "struct/class op"],
    [r"ret <%.*", "struct/class op", "struct/class op"],
    [r"ret <?{.*", "struct/class op", "struct/class op"],
    [r"ret opaque.*", "struct/class op", "struct/class op"],
    [r"ret <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    [r"ret <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    [r"ret <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    [r"ret <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    [r"ret <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    [r"ret <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    [r"ret <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    [r"ret <\d+ x i128> .*", "<d x i128> operation", "<d x int> operation"],
    [
        r"ret <\d+ x x86_fp80> .*",
        r"<d x x86_fp80> operation",
        r"<d x floating point> operation",
    ],
    [
        r"ret <\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"ret <\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [r"ret <\d+ x i1>\* .*", "<d x i1>*  operation", "<d x int>* operation"],
    [r"ret <\d+ x i2>\* .*", "<d x i2>*  operation", "<d x int>* operation"],
    [r"ret <\d+ x i4>\* .*", "<d x i4>*  operation", "<d x int>* operation"],
    [r"ret <\d+ x i8>\* .*", "<d x i8>*  operation", "<d x int>* operation"],
    [r"ret <\d+ x i16>\* .*", "<d x i16>* operation", "<d x int>* operation"],
    [r"ret <\d+ x i32>\* .*", "<d x i32>* operation", "<d x int>* operation"],
    [r"ret <\d+ x i64>\* .*", "<d x i64>* operation", "<d x int>* operation"],
    [r"ret <\d+ x i128>\* .*", "<d x i128>* operation", "<d x int>* operation"],
    [
        r"ret <\d+ x x86_fp80>\* .*",
        r"<d x x86_fp80>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"ret <\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"ret <\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [r"ret \[\d+ x i1\] .*", "[d x i1]  operation", "[d x int] operation"],
    [r"ret \[\d+ x i2\] .*", "[d x i2]  operation", "[d x int] operation"],
    [r"ret \[\d+ x i4\] .*", "[d x i4]  operation", "[d x int] operation"],
    [r"ret \[\d+ x i8\] .*", "[d x i8]  operation", "[d x int] operation"],
    [r"ret \[\d+ x i16\] .*", "[d x i16] operation", "[d x int] operation"],
    [r"ret \[\d+ x i32\] .*", "[d x i32] operation", "[d x int] operation"],
    [r"ret \[\d+ x i64\] .*", "[d x i64] operation", "[d x int] operation"],
    [r"ret \[\d+ x i128\] .*", "[d x i128] operation", "[d x int] operation"],
    [
        r"ret \[\d+ x x86_fp80\] .*",
        r"[d x x86_fp80] operation",
        r"[d x floating point] operation",
    ],
    [
        r"ret \[\d+ x float\] .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"ret \[\d+ x double\] .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [r"<%ID> = and i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = and <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    [r"<%ID> = and i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = and <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    [r"<%ID> = and i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = and <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    [r"<%ID> = and i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = and <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    [r"<%ID> = and i16 .*", "i16 operation", "int operation"],
    [r"<%ID> = and <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    [r"<%ID> = and i24 .*", "i24 operation", "int operation"],
    [r"<%ID> = and <\d+ x i24> .*", "<d x i24> operation", "<d x int> operation"],
    [r"<%ID> = and i32 .*", "i32 operation", "int operation"],
    [r"<%ID> = and <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    [r"<%ID> = and i40 .*", "i40 operation", "int operation"],
    [r"<%ID> = and <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    [r"<%ID> = and i64 .*", "i64 operation", "int operation"],
    [r"<%ID> = and <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    [r"<%ID> = and i128 .*", "i128 operation", "int operation"],
    [
        r"<%ID> = and <\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = or i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = or <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    [r"<%ID> = or i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = or <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    [r"<%ID> = or i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = or <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    [r"<%ID> = or i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = or <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    [r"<%ID> = or i16 .*", "i16 operation", "int operation"],
    [r"<%ID> = or <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    [r"<%ID> = or i24 .*", "i24 operation", "int operation"],
    [r"<%ID> = or <\d+ x i24> .*", "<d x i24> operation", "<d x int> operation"],
    [r"<%ID> = or i32 .*", "i32 operation", "int operation"],
    [r"<%ID> = or <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    [r"<%ID> = or i40 .*", "i40 operation", "int operation"],
    [r"<%ID> = or <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    [r"<%ID> = or i64 .*", "i64 operation", "int operation"],
    [r"<%ID> = or <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    [r"<%ID> = or i128 .*", "i128 operation", "int operation"],
    [r"<%ID> = or <\d+ x i128> .*", "<d x i128> operation", "<d x int> operation"],
    [r"<%ID> = xor i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = xor <\d+ x i1>.*", "<d x i1> operation", "<d x int> operation"],
    [r"<%ID> = xor i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = xor <\d+ x i2>.*", "<d x i2> operation", "<d x int> operation"],
    [r"<%ID> = xor i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = xor <\d+ x i4>.*", "<d x i4> operation", "<d x int> operation"],
    [r"<%ID> = xor i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = xor <\d+ x i8>.*", "<d x i8> operation", "<d x int> operation"],
    [r"<%ID> = xor i16 .*", "i16 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i16>.*", "<d x i16> operation", "<d x int> operation"],
    [r"<%ID> = xor i24 .*", "i16 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i24>.*", "<d x i16> operation", "<d x int> operation"],
    [r"<%ID> = xor i32 .*", "i32 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i32>.*", "<d x i32> operation", "<d x int> operation"],
    [r"<%ID> = xor i40 .*", "i40 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i40>.*", "<d x i40> operation", "<d x int> operation"],
    [r"<%ID> = xor i64 .*", "i64 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i64>.*", "<d x i64> operation", "<d x int> operation"],
    [r"<%ID> = xor i128 .*", "i128 operation", "int operation"],
    [r"<%ID> = xor <\d+ x i128>.*", "<d x i128> operation", "<d x int> operation"],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i4 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i40 .*",
        r"i40 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i40> .*",
        r"<d x i40> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?i256 .*",
        r"i256 operation",
        r"int operation",
    ],
    [
        r"<%ID> = shl " + any_of(opt_addsubmul) + r"?<\d+ x i256> .*",
        r"<d x i256> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i1 .*",
        r"i1 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i2 .*",
        r"i2 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i40 .*",
        r"i40 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i40> .*",
        r"<d x i40> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?i256 .*",
        r"i256 operation",
        r"int operation",
    ],
    [
        r"<%ID> = ashr " + any_of(opt_usdiv) + r"?<\d+ x i256> .*",
        r"<d x i256> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i1> .*",
        r"<d x i1> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i2> .*",
        r"<d x i2> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i4> .*",
        r"<d x i4> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i8> .*",
        r"<d x i8> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i24 .*",
        r"i24 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i24> .*",
        r"<d x i24> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i40 .*",
        r"i40 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i40> .*",
        r"<d x i40> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?i256 .*",
        r"i256 operation",
        r"int operation",
    ],
    [
        r"<%ID> = lshr " + any_of(opt_usdiv) + r"?<\d+ x i256> .*",
        r"<d x i256> operation",
        r"<d x int> operation",
    ],
    [r"<%ID> = phi i1 .*", "i1  operation", "int operation"],
    [r"<%ID> = phi <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i1\*> .*",
        r"<d x i1*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i1>\* .*",
        r"<d x i1>* operation",
        r"<d x int>* operation",
    ],
    [r"<%ID> = phi \[\d+ x i1\] .*", "[d x i1] operation", "[d x int] operation"],
    [
        r"<%ID> = phi \[\d+ x i1\]\* .*",
        r"[d x i1]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i1\]\*\* .*",
        r"[d x i1]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i1\]\*\*\* .*",
        r"[d x i1]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i2 .*", "i2  operation", "int operation"],
    [r"<%ID> = phi <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i2\*> .*",
        r"<d x i2*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i2>\* .*",
        r"<d x i2>* operation",
        r"<d x int>* operation",
    ],
    [r"<%ID> = phi \[\d+ x i2\] .*", "[d x i2] operation", "[d x int] operation"],
    [
        r"<%ID> = phi \[\d+ x i2\]\* .*",
        r"[d x i2]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i2\]\*\* .*",
        r"[d x i2]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i2\]\*\*\* .*",
        r"[d x i2]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i4 .*", "i4  operation", "int operation"],
    [r"<%ID> = phi <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i4\*> .*",
        r"<d x i4*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i4>\* .*",
        r"<d x i4>* operation",
        r"<d x int>* operation",
    ],
    [r"<%ID> = phi \[\d+ x i4\] .*", "[d x i4] operation", "[d x int] operation"],
    [
        r"<%ID> = phi \[\d+ x i4\]\* .*",
        r"[d x i4]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i4\]\*\* .*",
        r"[d x i4]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i4\]\*\*\* .*",
        r"[d x i4]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i8 .*", "i8  operation", "int operation"],
    [r"<%ID> = phi <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i8\*> .*",
        r"<d x i8*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i8>\* .*",
        r"<d x i8>* operation",
        r"<d x int>* operation",
    ],
    [r"<%ID> = phi \[\d+ x i8\] .*", "[d x i4] operation", "[d x int] operation"],
    [
        r"<%ID> = phi \[\d+ x i8\]\* .*",
        r"[d x i4]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i8\]\*\* .*",
        r"[d x i4]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i8\]\*\*\* .*",
        r"[d x i4]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i16 .*", "i16 operation", "int operation"],
    [r"<%ID> = phi <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i16\*> .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i16>\* .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i16\] .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i16\]\* .*",
        r"[d x i16]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i16\]\*\* .*",
        r"[d x i16]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i16\]\*\*\* .*",
        r"[d x i16]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i32 .*", "i32 operation", "int operation"],
    [r"<%ID> = phi <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i32\*> .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i32>\* .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i32\] .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i32\]\* .*",
        r"[d x i32]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i32\]\*\* .*",
        r"[d x i32]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i32\]\*\*\* .*",
        r"[d x i32]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i40 .*", "i32 operation", "int operation"],
    [r"<%ID> = phi <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i40\*> .*",
        r"<d x i40*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i40>\* .*",
        r"<d x i40>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i40\] .*",
        r"[d x i40] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i40\]\* .*",
        r"[d x i40]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i40\]\*\* .*",
        r"[d x i40]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i40\]\*\*\* .*",
        r"[d x i40]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i64 .*", "i64 operation", "int operation"],
    [r"<%ID> = phi <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    [
        r"<%ID> = phi <\d+ x i64\*> .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i64>\* .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i64\] .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i64\]\* .*",
        r"[d x i64]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i64\]\*\* .*",
        r"[d x i64]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i64\]\*\*\* .*",
        r"[d x i64]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i128 .*", "i128 operation", "int operation"],
    [
        r"<%ID> = phi <\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i128\*> .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x i128>\* .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i128\] .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i128\]\* .*",
        r"[d x i128]* operation",
        r"[d x int]* operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i126\]\*\* .*",
        r"[d x i128]** operation",
        r"[d x int]** operation",
    ],
    [
        r"<%ID> = phi \[\d+ x i128\]\*\*\* .*",
        r"[d x i128]*** operation",
        r"[d x int]*** operation",
    ],
    [r"<%ID> = phi i1\* .*", "i1*  operation", "int* operation"],
    [r"<%ID> = phi i2\* .*", "i2*  operation", "int* operation"],
    [r"<%ID> = phi i4\* .*", "i4*  operation", "int* operation"],
    [r"<%ID> = phi i8\* .*", "i8*  operation", "int* operation"],
    [r"<%ID> = phi i16\* .*", "i16*  operation", "int* operation"],
    [r"<%ID> = phi i32\* .*", "i32*  operation", "int* operation"],
    [r"<%ID> = phi i40\* .*", "i40*  operation", "int* operation"],
    [r"<%ID> = phi i64\* .*", "i64*  operation", "int* operation"],
    [r"<%ID> = phi i128\* .*", "i128*  operation", "int* operation"],
    [r"<%ID> = phi i1\*\* .*", "i1**  operation", "int** operation"],
    [r"<%ID> = phi i2\*\* .*", "i2**  operation", "int** operation"],
    [r"<%ID> = phi i4\*\* .*", "i4**  operation", "int** operation"],
    [r"<%ID> = phi i8\*\* .*", "i8**  operation", "int** operation"],
    [r"<%ID> = phi i16\*\* .*", "i16** operation", "int** operation"],
    [r"<%ID> = phi i32\*\* .*", "i32** operation", "int** operation"],
    [r"<%ID> = phi i40\*\* .*", "i40** operation", "int** operation"],
    [r"<%ID> = phi i64\*\* .*", "i64** operation", "int** operation"],
    [r"<%ID> = phi i128\*\* .*", "i128** operation", "int** operation"],
    [r"<%ID> = phi i1\*\*\* .*", "i1***  operation", "int*** operation"],
    [r"<%ID> = phi i2\*\*\* .*", "i2***  operation", "int*** operation"],
    [r"<%ID> = phi i4\*\*\* .*", "i4***  operation", "int*** operation"],
    [r"<%ID> = phi i8\*\*\* .*", "i8***  operation", "int*** operation"],
    [r"<%ID> = phi i16\*\*\* .*", "i16*** operation", "int*** operation"],
    [r"<%ID> = phi i32\*\*\* .*", "i32*** operation", "int*** operation"],
    [r"<%ID> = phi i64\*\*\* .*", "i64*** operation", "int*** operation"],
    [r"<%ID> = phi i128\*\*\* .*", "i128*** operation", "int*** operation"],
    [r"<%ID> = phi x86_fp80 .*", "float  operation", "floating point operation"],
    [r"<%ID> = phi float .*", "float  operation", "floating point operation"],
    [r"<%ID> = phi double .*", "double operation", "floating point operation"],
    [
        r"<%ID> = phi <\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = phi <\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = phi <\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = phi x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = phi <\d+ x x86_fp80\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x float\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x double\*> .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = phi <\d+ x x86_fp80>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = phi <\d+ x float>\* .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = phi <\d+ x double>\* .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = phi x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [r"<%ID> = phi float\* .*", "float*  operation", "floating point* operation"],
    [r"<%ID> = phi double\* .*", "double* operation", "floating point* operation"],
    [
        r"<%ID> = phi x86_fp80\*\* .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = phi float\*\* .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = phi double\*\* .*",
        r"double**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = phi x86_fp80\*\*\* .*",
        r"float***  operation",
        r"floating point*** operation",
    ],
    [
        r"<%ID> = phi float\*\*\* .*",
        r"float***  operation",
        r"floating point*** operation",
    ],
    [
        r"<%ID> = phi double\*\*\* .*",
        r"double***  operation",
        r"floating point*** operation",
    ],
    [r"<%ID> = phi void \(.*\) \[.*", "function op", "function op"],
    [r"<%ID> = phi void \(.*\)\* \[.*", "function* op", "function* op"],
    [r"<%ID> = phi void \(.*\)\*\* \[.*", "function** op", "function** op"],
    [r"<%ID> = phi void \(.*\)\*\*\* \[.*", "function*** op", "function*** op"],
    [r"<%ID> = phi (<?{|opaque|<%ID>) .*", "struct/class op", "struct/class op"],
    [
        r"<%ID> = phi (<?{|opaque|<%ID>)\* .*",
        r"struct/class* op",
        r"struct/class* op",
    ],
    [
        r"<%ID> = phi (<?{|opaque|<%ID>)\*\* .*",
        r"struct/class** op",
        r"struct/class** op",
    ],
    [
        r"<%ID> = phi (<?{|opaque|<%ID>)\*\*\* .*",
        r"struct/class*** op",
        r"struct/class*** op",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i1, .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i2, .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i4, .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i8, .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i16, .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i32, .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i64, .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i128, .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i1\*, .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i2\*, .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i4\*, .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i8\*, .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i16\*, .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i32\*, .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i64\*, .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i128\*, .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i1\*\*, .*",
        r"i1**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i2\*\*, .*",
        r"i2**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i4\*\*, .*",
        r"i4**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i8\*\*, .*",
        r"i8**  operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i16\*\*, .*",
        r"i16** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i32\*\*, .*",
        r"i32** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i64\*\*, .*",
        r"i64** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"i128\*\*, .*",
        r"i128** operation",
        r"int** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"x86_fp80, .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"float, .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"double, .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"x86_fp80\*, .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"float\*, .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"double\*, .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"x86_fp80\*\*, .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"float\*\*, .*",
        r"float**  operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"double\*\*, .*",
        r"double** operation",
        r"floating point** operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r'%".*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<%.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i1>, .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i2>, .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i4>, .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i8>, .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i16>, .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i32>, .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i64>, .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i128>, .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x x86_fp80>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x float>, .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x double>, .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i1>\*, .*",
        r"<d x i1>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i2>\*, .*",
        r"<d x i2>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i4>\*, .*",
        r"<d x i4>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i8>\*, .*",
        r"<d x i8>*  operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i16>\*, .*",
        r"<d x i16>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i32>\*, .*",
        r"<d x i32>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i64>\*, .*",
        r"<d x i64>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x i128>\*, .*",
        r"<d x i128>* operation",
        r"<d x int>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x x86_fp80>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x float>\*, .*",
        r"<d x float>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"<\d+ x double>\*, .*",
        r"<d x double>* operation",
        r"<d x floating point>* operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i1\], .*",
        r"[d x i1]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i2\], .*",
        r"[d x i2]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i4\], .*",
        r"[d x i4]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i8\], .*",
        r"[d x i8]  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i16\], .*",
        r"[d x i16] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i32\], .*",
        r"[d x i32] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i64\], .*",
        r"[d x i64] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i128\], .*",
        r"[d x i128] operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x x86_fp80\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x float\], .*",
        r"[d x float] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x double\], .*",
        r"[d x double] operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x .*\], .*",
        r"array of array operation",
        r"array of array operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i1\]\*, .*",
        r"[d x i1]*  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i2\]\*, .*",
        r"[d x i2]*  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i4\]\*, .*",
        r"[d x i4]*  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i8\]\*, .*",
        r"[d x i8]*  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i16\]\*, .*",
        r"[d x i16]* operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i32\]\*, .*",
        r"[d x i32]* operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i64\]\*, .*",
        r"[d x i64]* operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i128\]\*, .*",
        r"[d x i128]* operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x x86_fp80\]\*, .*",
        r"[d x float]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x float\]\*, .*",
        r"[d x float]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x double\]\*, .*",
        r"[d x double]* operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x .*\]\*, .*",
        r"array of array* operation",
        r"array of array operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i1\]\*\*, .*",
        r"[d x i1]**  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i2\]\*\*, .*",
        r"[d x i2]**  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i4\]\*\*, .*",
        r"[d x i4]**  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i8\]\*\*, .*",
        r"[d x i8]**  operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i16\]\*\*, .*",
        r"[d x i16]** operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i32\]\*\*, .*",
        r"[d x i32]** operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i64\]\*\*, .*",
        r"[d x i64]** operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x i128\]\*\*, .*",
        r"[d x i128]** operation",
        r"[d x int] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x x86_fp80\]\*\*, .*",
        r"[d x float]** operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x float\]\*\*, .*",
        r"[d x float]** operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x double\]\*\*, .*",
        r"[d x double]** operation",
        r"[d x floating point] operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r"\[\d+ x .*\], .*",
        r"array of array** operation",
        r"array of array operation",
    ],
    [
        r"<%ID> = getelementptr " + any_of(opt_GEP) + r".*\(.*\)\*+, .*",
        r"function operation",
        r"function operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i1 .*",
        r"i1  operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i2 .*",
        r"i2  operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i4 .*",
        r"i4  operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i8 .*",
        r"i8  operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i16 .*",
        r"i16 operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i32 .*",
        r"i32 operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i64 .*",
        r"i64 operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i128 .*",
        r"i128 operation",
        r"int operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i1\* .*",
        r"i1*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i2\* .*",
        r"i2*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i4\* .*",
        r"i4*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i8\* .*",
        r"i8*  operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i16\* .*",
        r"i16* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i32\* .*",
        r"i32* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i64\* .*",
        r"i64* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*i128\* .*",
        r"i128* operation",
        r"int* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*x86_fp80 .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*float .*",
        r"float  operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*double .*",
        r"double operation",
        r"floating point operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*x86_fp80\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*float\* .*",
        r"float*  operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*double\* .*",
        r"double* operation",
        r"floating point* operation",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r'*%".*',
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*<?{.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r"*opaque.*",
        r"struct/class op",
        r"struct/class op",
    ],
    [
        r"<%ID> = invoke " + any_of(opt_invoke) + r'*%".*\*.*',
        r"struct/class* op",
        r"struct/class op",
    ],
    [r"<%ID> = invoke " + any_of(opt_invoke) + r"*void .*", "void op", "void op"],
    [r"invoke " + any_of(opt_invoke) + r"*void .*", "void op", "void op"],
    [
        r"<%ID> = extractelement <\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i1\*> .*",
        r"<d x i1*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i2\*> .*",
        r"<d x i2*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i4\*> .*",
        r"<d x i4*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i8\*> .*",
        r"<d x i8*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i16\*> .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i32\*> .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i64\*> .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x i128\*> .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x x86_fp80\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x float\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x double\*> .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x \{.*\}> .*",
        r"<d x struct> operation",
        r"<d x struct> operation",
    ],
    [
        r"<%ID> = extractelement <\d+ x \{.*\}\*> .*",
        r"<d x struct*> operation",
        r"<d x struct*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i1\*> .*",
        r"<d x i1*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i2\*> .*",
        r"<d x i2*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i4\*> .*",
        r"<d x i4*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i8\*> .*",
        r"<d x i8*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i16\*> .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i32\*> .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i64\*> .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x i128\*> .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x x86_fp80\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x float\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x double\*> .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x \{.*\}> .*",
        r"<d x struct> operation",
        r"<d x struct> operation",
    ],
    [
        r"<%ID> = insertelement <\d+ x \{.*\}\*> .*",
        r"<d x struct*> operation",
        r"<d x struct*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i1> .*",
        r"<d x i1>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i1\*> .*",
        r"<d x i1*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i2> .*",
        r"<d x i2>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i2\*> .*",
        r"<d x i2*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i4> .*",
        r"<d x i4>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i4\*> .*",
        r"<d x i4*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i8> .*",
        r"<d x i8>  operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i8\*> .*",
        r"<d x i8*>  operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i16> .*",
        r"<d x i16> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i16\*> .*",
        r"<d x i16*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i32> .*",
        r"<d x i32> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i32\*> .*",
        r"<d x i32*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i64> .*",
        r"<d x i64> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i64\*> .*",
        r"<d x i64*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i128> .*",
        r"<d x i128> operation",
        r"<d x int> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x i128\*> .*",
        r"<d x i128*> operation",
        r"<d x int*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x x86_fp80> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x x86_fp80\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x float> .*",
        r"<d x float> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x float\*> .*",
        r"<d x float*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x double> .*",
        r"<d x double> operation",
        r"<d x floating point> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x double\*> .*",
        r"<d x double*> operation",
        r"<d x floating point*> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x \{.*\}> .*",
        r"<d x struct> operation",
        r"<d x struct> operation",
    ],
    [
        r"<%ID> = shufflevector <\d+ x \{.*\}\*> .*",
        r"<d x struct*> operation",
        r"<d x struct*> operation",
    ],
    [
        r"<%ID> = bitcast void \(.* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque) .* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\* .* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\* .* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\* .* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast \[\d+.* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r"<%ID> = bitcast <\d+.* to .*",
        r"in-between operation",
        r"in-between operation",
    ],
    [
        r'<%ID> = bitcast (%"|<%|<?{).* to .*',
        r"in-between operation",
        r"in-between operation",
    ],
    [r"<%ID> = fpext .*", "in-between operation", "in-between operation"],
    [r"<%ID> = fptrunc .*", "in-between operation", "in-between operation"],
    [r"<%ID> = sext .*", "in-between operation", "in-between operation"],
    [r"<%ID> = trunc .* to .*", "in-between operation", "in-between operation"],
    [r"<%ID> = zext .*", "in-between operation", "in-between operation"],
    [r"<%ID> = sitofp .*", "in-between operation", "in-between operation"],
    [r"<%ID> = uitofp .*", "in-between operation", "in-between operation"],
    [r"<%ID> = inttoptr .*", "in-between operation", "in-between operation"],
    [r"<%ID> = ptrtoint .*", "in-between operation", "in-between operation"],
    [r"<%ID> = fptosi .*", "in-between operation", "in-between operation"],
    [r"<%ID> = fptoui .*", "in-between operation", "in-between operation"],
    [r"<%ID> = extractvalue .*", "in-between operation", "in-between operation"],
    [r"<%ID> = insertvalue .*", "in-between operation", "in-between operation"],
    [r"resume .*", "in-between operation", "in-between operation"],
    [r"(tail |musttail |notail )?call( \w+)? void .*", "call void", "call void"],
    [r"i\d{1,2} <(INT|FLOAT)>, label <%ID>", "blob", "blob"],
    [r"<%ID> = select .*", "blob", "blob"],
    [r".*to label.*unwind label.*", "blob", "blob"],
    [r"catch .*", "blob", "blob"],
    [r"cleanup", "blob", "blob"],
    [r"<%ID> = landingpad .", "blob", "blob"],
    [r"; <label>:<LABEL>", "blob", "blob"],
    [r"<LABEL>:", "blob", "blob"],
    [r"br i1 .*", "blob", "blob"],
    [r"br label .*", "blob", "blob"],
    [r"indirectbr .*", "blob", "blob"],
    [r"switch .*", "blob", "blob"],
    [r"unreachable.*", "blob", "blob"],
    [r"ret void", "blob", "blob"],
    [r"!UNK", "blob", "blob"],
]
