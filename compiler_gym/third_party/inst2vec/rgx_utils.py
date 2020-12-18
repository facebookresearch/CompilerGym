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
"""Helper variables and functions for regular expressions and statement tags"""
import re


########################################################################################################################
# Regex manipulation: helper functions
########################################################################################################################
def any_of(possibilities, to_add=""):
    """
    Helper function for regex manipulation:
    Construct a regex representing "any of" the given possibilities
    :param possibilities: list of strings representing different word possibilities
    :param to_add: string to add at the beginning of each possibility (optional)
    :return: string corresponding to regex which represents any of the given possibilities
    """
    assert len(possibilities) > 0
    s = "(?:"
    if len(to_add) > 0:
        s += possibilities[0] + to_add + " "
    else:
        s += possibilities[0]
    for i in range(len(possibilities) - 1):
        if len(to_add) > 0:
            s += "|" + possibilities[i + 1] + to_add + " "
        else:
            s += "|" + possibilities[i + 1]
    return s + ")"


########################################################################################################################
# Regex manipulation: helper variables
########################################################################################################################
# Identifiers
global_id = r'(?<!%")@["\w\d\.\-\_\$\\]+'
local_id_no_perc = '["\@\d\w\.\-\_\:]+'
local_id = "%" + local_id_no_perc
local_or_global_id = r"(" + global_id + r"|" + local_id + r")"

# Options and linkages
linkage = any_of(
    [
        " private",
        " external",
        " internal",
        " linkonce_odr",
        " appending",
        " external",
        " internal",
        " unnamed_addr",
        " common",
        " hidden",
        " weak",
        " linkonce",
        " extern_weak",
        " weak_odr",
        " private",
        " available_externally",
        " local_unnamed_addr",
        " thread_local",
        " linker_private",
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
    r"(?:" + immediate_value_float_sci + "|" + immediate_value_float_hexa + ")"
)
immediate_value_vector_bool = (
    r"<i1 "
    + immediate_value_bool
    + r"(?:, i1 (?:"
    + immediate_value_bool
    + "|undef))*>"
)
immediate_value_vector_int = (
    r"<i\d+ "
    + immediate_value_int
    + r"(?:, i\d+ (?:"
    + immediate_value_int
    + "|undef))*>"
)
immediate_value_vector_float = (
    r"<float "
    + immediate_value_float
    + r"(?:, float (?:"
    + immediate_value_float
    + "|undef))*>"
)
immediate_value_vector_double = (
    r"<double "
    + immediate_value_float
    + r"(?:, double (?:"
    + immediate_value_float
    + "|undef))*>"
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
struct_name_add_on = '(?:\([\w\d=]+\)")?'
struct_name_without_lookahead = (
    '%["\@\d\w\.\-\_:]+(?:(?:<["\@\d\w\.\-\_:,<>\(\) \*]+>|\(["\@\d\w\.\-\_:,<> \*]+\)|\w+)?::[" \@\d\w\.\-\_:\)\(]*)*'
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
    "i\d+",
    "half",
    "float",
    "double",
    "fp_128",
    "x86_fp80",
    "ppc_fp128",
    "<%ID>",
]
first_class_type = any_of(first_class_types) + "\**"
base_type_or_struct_name = any_of([base_type, struct_name_without_lookahead])
ptr_to_base_type = base_type + r"\*+"
vector_type = r"<\d+ x " + base_type + r">"
ptr_to_vector_type = vector_type + r"\*+"
array_type = r"\[\d+ x " + base_type + r"\]"
ptr_to_array_type = array_type + r"\*+"
array_of_array_type = "\[\d+ x " + "\[\d+ x " + base_type + "\]" + "\]"
struct = struct_name_without_lookahead
ptr_to_struct = struct + r"\*+"
function_type = (
    base_type
    + " \("
    + any_of([base_type, vector_type, array_type, "..."], ",")
    + "*"
    + any_of([base_type, vector_type, array_type, "..."])
    + "\)\**"
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
    "(<?{ " + structure_entry_with_comma + "*" + structure_entry + " }>?|{})"
)

# Tokens
unknown_token = "!UNK"  # starts with '!' to guarantee it will appear first in the alphabetically sorted vocabulary

########################################################################################################################
# Tags for clustering statements (by statement semantics) and helper functions
########################################################################################################################
# List of families of operations
llvm_IR_stmt_families = [
    # ["tag level 1",                  "tag level 2",            "tag level 3",              "regex"                    ]
    ["unknown token", "unknown token", "unknown token", "!UNK"],
    ["integer arithmetic", "addition", "add integers", "<%ID> = add .*"],
    ["integer arithmetic", "subtraction", "subtract integers", "<%ID> = sub .*"],
    [
        "integer arithmetic",
        "multiplication",
        "multiply integers",
        "<%ID> = mul .*",
    ],
    [
        "integer arithmetic",
        "division",
        "unsigned integer division",
        "<%ID> = udiv .*",
    ],
    [
        "integer arithmetic",
        "division",
        "signed integer division",
        "<%ID> = sdiv .*",
    ],
    [
        "integer arithmetic",
        "remainder",
        "remainder of signed div",
        "<%ID> = srem .*",
    ],
    [
        "integer arithmetic",
        "remainder",
        "remainder of unsigned div.",
        "<%ID> = urem .*",
    ],
    ["floating-point arithmetic", "addition", "add floats", "<%ID> = fadd .*"],
    [
        "floating-point arithmetic",
        "subtraction",
        "subtract floats",
        "<%ID> = fsub .*",
    ],
    [
        "floating-point arithmetic",
        "multiplication",
        "multiply floats",
        "<%ID> = fmul .*",
    ],
    ["floating-point arithmetic", "division", "divide floats", "<%ID> = fdiv .*"],
    ["bitwise arithmetic", "and", "and", "<%ID> = and .*"],
    ["bitwise arithmetic", "or", "or", "<%ID> = or .*"],
    ["bitwise arithmetic", "xor", "xor", "<%ID> = xor .*"],
    ["bitwise arithmetic", "shift left", "shift left", "<%ID> = shl .*"],
    ["bitwise arithmetic", "arithmetic shift right", "ashr", "<%ID> = ashr .*"],
    [
        "bitwise arithmetic",
        "logical shift right",
        "logical shift right",
        "<%ID> = lshr .*",
    ],
    [
        "comparison operation",
        "compare integers",
        "compare integers",
        "<%ID> = icmp .*",
    ],
    [
        "comparison operation",
        "compare floats",
        "compare floats",
        "<%ID> = fcmp .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast single val",
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque) .* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast single val*",
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\* .* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast single val**",
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\* .* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast single val***",
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\* .* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast single val****",
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\*\* .* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast array",
        "<%ID> = bitcast \[\d.* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast vector",
        "<%ID> = bitcast <\d.* to .*",
    ],
    [
        "conversion operation",
        "bitcast",
        "bitcast structure",
        '<%ID> = bitcast (%"|<{|<%|{).* to .*',
    ],
    ["conversion operation", "bitcast", "bitcast void", "<%ID> = bitcast void "],
    [
        "conversion operation",
        "extension/truncation",
        "extend float",
        "<%ID> = fpext .*",
    ],
    [
        "conversion operation",
        "extension/truncation",
        "truncate floats",
        "<%ID> = fptrunc .*",
    ],
    [
        "conversion operation",
        "extension/truncation",
        "sign extend ints",
        "<%ID> = sext .*",
    ],
    [
        "conversion operation",
        "extension/truncation",
        "truncate int to ... ",
        "<%ID> = trunc .* to .*",
    ],
    [
        "conversion operation",
        "extension/truncation",
        "zero extend integers",
        "<%ID> = zext .*",
    ],
    [
        "conversion operation",
        "convert",
        "convert signed integers to... ",
        "<%ID> = sitofp .*",
    ],
    [
        "conversion operation",
        "convert",
        "convert unsigned integer to... ",
        "<%ID> = uitofp .*",
    ],
    [
        "conversion operation",
        "convert int to ptr",
        "convert int to ptr",
        "<%ID> = inttoptr .*",
    ],
    [
        "conversion operation",
        "convert ptr to int",
        "convert ptr to int",
        "<%ID> = ptrtoint .*",
    ],
    [
        "conversion operation",
        "convert floats",
        "convert float to sint",
        "<%ID> = fptosi .*",
    ],
    [
        "conversion operation",
        "convert floats",
        "convert float to uint",
        "<%ID> = fptoui .*",
    ],
    ["control flow", "phi", "phi", "<%ID> = phi .*"],
    [
        "control flow",
        "switch",
        "jump table line",
        "i\d{1,2} <(INT|FLOAT)>, label <%ID>",
    ],
    ["control flow", "select", "select", "<%ID> = select .*"],
    ["control flow", "invoke", "invoke and ret type", "<%ID> = invoke .*"],
    ["control flow", "invoke", "invoke void", "invoke (fastcc )?void .*"],
    ["control flow", "branch", "branch conditional", "br i1 .*"],
    ["control flow", "branch", "branch unconditional", "br label .*"],
    ["control flow", "branch", "branch indirect", "indirectbr .*"],
    ["control flow", "control flow", "switch", "switch .*"],
    ["control flow", "return", "return", "ret .*"],
    ["control flow", "resume", "resume", "resume .*"],
    ["control flow", "unreachable", "unreachable", "unreachable.*"],
    ["control flow", "exception handling", "catch block", "catch .*"],
    ["control flow", "exception handling", "cleanup clause", "cleanup"],
    [
        "control flow",
        "exception handling",
        "landingpad for exceptions",
        "<%ID> = landingpad .",
    ],
    [
        "function",
        "function call",
        "sqrt (llvm-intrinsic)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>) @(llvm|llvm\..*)\.sqrt.*",
    ],
    [
        "function",
        "function call",
        "fabs (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fabs.*",
    ],
    [
        "function",
        "function call",
        "max (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.max.*",
    ],
    [
        "function",
        "function call",
        "min (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.min.*",
    ],
    [
        "function",
        "function call",
        "fma (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fma.*",
    ],
    [
        "function",
        "function call",
        "phadd (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.phadd.*",
    ],
    [
        "function",
        "function call",
        "pabs (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pabs.*",
    ],
    [
        "function",
        "function call",
        "pmulu (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pmulu.*",
    ],
    [
        "function",
        "function call",
        "umul (llvm-intr.)",
        "<%ID> = (tail |musttail |notail )?call {.*} @llvm\.umul.*",
    ],
    [
        "function",
        "function call",
        "prefetch (llvm-intr.)",
        "(tail |musttail |notail )?call void @llvm\.prefetch.*",
    ],
    [
        "function",
        "function call",
        "trap (llvm-intr.)",
        "(tail |musttail |notail )?call void @llvm\.trap.*",
    ],
    ["function", "func decl / def", "function declaration", "declare .*"],
    ["function", "func decl / def", "function definition", "define .*"],
    [
        "function",
        "function call",
        "function call void",
        "(tail |musttail |notail )?call( \w+)? void [\w\)\(\}\{\.\,\*\d\[\]\s<>%]*(<[@%]ID>\(|.*bitcast )",
    ],
    [
        "function",
        "function call",
        "function call mem lifetime",
        "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.lifetime.*",
    ],
    [
        "function",
        "function call",
        "function call mem copy",
        "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memcpy\..*",
    ],
    [
        "function",
        "function call",
        "function call mem set",
        "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memset\..*",
    ],
    [
        "function",
        "function call",
        "function call single val",
        "<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80|<\d+ x (i\d+|float|double)>) (.*<[@%]ID>\(|(\(.*\) )?bitcast ).*",
    ],
    [
        "function",
        "function call",
        "function call single val*",
        "<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\* (.*<[@%]ID>\(|\(.*\) bitcast ).*",
    ],
    [
        "function",
        "function call",
        "function call single val**",
        "<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\*\* (.*<[@%]ID>\(|\(.*\) bitcast ).*",
    ],
    [
        "function",
        "function call",
        "function call array",
        "<%ID> = (tail |musttail |notail )?call[^{]* \[.*\] (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call array*",
        "<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call array**",
        "<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\*\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call structure",
        "<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>) (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call structure*",
        "<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call structure**",
        "<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call structure***",
        "<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )",
    ],
    [
        "function",
        "function call",
        "function call asm value",
        "<%ID> = (tail |musttail |notail )?call.* asm .*",
    ],
    [
        "function",
        "function call",
        "function call asm void",
        "(tail |musttail |notail )?call void asm .*",
    ],
    [
        "function",
        "function call",
        "function call function",
        "<%ID> = (tail |musttail |notail )?call[^{]* void \([^\(\)]*\)\** <[@%]ID>\(",
    ],
    [
        "global variables",
        "glob. var. definition",
        "???",
        "<@ID> = (?!.*constant)(?!.*alias).*",
    ],
    ["global variables", "constant definition", "???", "<@ID> = .*constant .*"],
    [
        "memory access",
        "load from memory",
        "load structure",
        '<%ID> = load (\w* )?(%"|<\{|\{ <|\{ \[|\{ |<%|opaque).*',
    ],
    [
        "memory access",
        "load from memory",
        "load single val",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val*",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val**",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val***",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val****",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val*****",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val******",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load single val*******",
        "<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*\*[, ].*",
    ],
    [
        "memory access",
        "load from memory",
        "load vector",
        "<%ID> = load <\d+ x .*",
    ],
    ["memory access", "load from memory", "load array", "<%ID> = load \[\d.*"],
    [
        "memory access",
        "load from memory",
        "load fction ptr",
        "<%ID> = load void \(",
    ],
    ["memory access", "store", "store", "store.*"],
    ["memory addressing", "GEP", "GEP", "<%ID> = getelementptr .*"],
    [
        "memory allocation",
        "allocate on stack",
        "allocate structure",
        '<%ID> = alloca (%"|<{|<%|{ |opaque).*',
    ],
    [
        "memory allocation",
        "allocate on stack",
        "allocate vector",
        "<%ID> = alloca <\d.*",
    ],
    [
        "memory allocation",
        "allocate on stack",
        "allocate array",
        "<%ID> = alloca \[\d.*",
    ],
    [
        "memory allocation",
        "allocate on stack",
        "allocate single value",
        "<%ID> = alloca (double|float|i\d{1,3})\*?.*",
    ],
    [
        "memory allocation",
        "allocate on stack",
        "allocate void",
        "<%ID> = alloca void \(.*",
    ],
    [
        "memory atomics",
        "atomic memory modify",
        "atomicrw xchg",
        "<%ID> = atomicrmw.* xchg .*",
    ],
    [
        "memory atomics",
        "atomic memory modify",
        "atomicrw add",
        "<%ID> = atomicrmw.* add .*",
    ],
    [
        "memory atomics",
        "atomic memory modify",
        "atomicrw sub",
        "<%ID> = atomicrmw.* sub .*",
    ],
    [
        "memory atomics",
        "atomic memory modify",
        "atomicrw or",
        "<%ID> = atomicrmw.* or .*",
    ],
    [
        "memory atomics",
        "atomic compare exchange",
        "cmpxchg single val",
        "<%ID> = cmpxchg (weak )?(i\d+|float|double|x86_fp80)\*",
    ],
    [
        "non-instruction",
        "label",
        "label declaration",
        "; <label>:.*(\s+; preds = <LABEL>)?",
    ],
    [
        "non-instruction",
        "label",
        "label declaration",
        "<LABEL>:( ; preds = <LABEL>)?",
    ],
    [
        "value aggregation",
        "extract value",
        "extract value",
        "<%ID> = extractvalue .*",
    ],
    [
        "value aggregation",
        "insert value",
        "insert value",
        "<%ID> = insertvalue .*",
    ],
    [
        "vector operation",
        "insert element",
        "insert element",
        "<%ID> = insertelement .*",
    ],
    [
        "vector operation",
        "extract element",
        "extract element",
        "<%ID> = extractelement .*",
    ],
    [
        "vector operation",
        "shuffle vector",
        "shuffle vector",
        "<%ID> = shufflevector .*",
    ],
]


# Helper functions for exploring llvm_IR_families
def get_list_tag_level_1():
    """
    Get the list of all level-1 tags in the data structure llvm_IR_families

    :return: list containing strings corresponding to all level 1 tags
    """
    list_tags = list()
    for fam in llvm_IR_stmt_families:
        list_tags.append(fam[0])

    return list(set(list_tags))


def get_list_tag_level_2(tag_level_1="all"):
    """
    Get the list of all level-2 tags in the data structure llvm_IR_families
    corresponding to the string given as an input, or absolutely all of them
    if input == 'all'

    :param tag_level_1: string containing the level-1 tag to query, or 'all'
    :return: list of strings
    """

    # Make sure the input parameter is valid
    assert tag_level_1 in get_list_tag_level_1() or tag_level_1 == "all", (
        tag_level_1 + " invalid"
    )

    list_tags = list()

    if tag_level_1 == "all":
        for fam in llvm_IR_stmt_families:
            list_tags.append(fam[1])
        list_tags = sorted(set(list_tags))
    else:
        for fam in llvm_IR_stmt_families:
            if fam[0] == tag_level_1:
                list_tags.append(fam[1])

    return list(set(list_tags))


def get_list_tag_level_3(tag_level_2="all"):
    """
    Get the list of all level-2 tags in the data structure llvm_IR_families
    corresponding to the string given as an input, or absolutely all of them
    if input == 'all'

    :param tag_level_2: string containing the level-2 tag to query, or 'all'
    :return: list of strings
    """

    # Make sure the input parameter is valid
    assert tag_level_2 in get_list_tag_level_2() or tag_level_2 == "all"

    list_tags = list()

    if tag_level_2 == "all":
        for fam in llvm_IR_stmt_families:
            list_tags.append(fam[2])
        list_tags = sorted(set(list_tags))
    else:
        for fam in llvm_IR_stmt_families:
            if fam[1] == tag_level_2:
                list_tags.append(fam[2])

    return list(set(list_tags))


def get_count(data, tag, level):
    """
    Count the total number of occurrences of a given tag at a certain level

    :param fams:
    :param tag:
    :param level:
    :return:
    """
    # Make sure the input is valid
    assert level in [1, 2, 3]

    # Count
    count = 0

    # Depending on tag level:
    if level == 1:
        assert tag in get_list_tag_level_1()
        for fam in llvm_IR_stmt_families:
            if fam[0] == tag:
                # count += fam[3]
                # count occurences in data
                for key, value in data.items():
                    if re.match(fam[3], key):
                        count += value
    elif level == 2:
        assert tag in get_list_tag_level_2()
        for fam in llvm_IR_stmt_families:
            if fam[1] == tag:
                # count += fam[3]
                # count occurences in data
                for key, value in data.items():
                    if re.match(fam[3], key):
                        count += value
    elif level == 3:
        assert tag in get_list_tag_level_3()
        for fam in llvm_IR_stmt_families:
            if fam[2] == tag:
                # count += fam[3]
                # count occurences in data
                for key, value in data.items():
                    if re.match(fam[3], key):
                        count += value

    return count


########################################################################################################################
# Tags for clustering statements (by statement type)
########################################################################################################################
# Helper lists
types_int = ["i1", "i8", "i16", "i32", "i64"]
types_flpt = ["half", "float", "double", "fp128", "x86_fp80", "ppc_fp128"]
fast_math_flag = [
    "",
    "nnan ",
    "ninf ",
    "nsz ",
    "arcp ",
    "contract ",
    "afn ",
    "reassoc ",
    "fast ",
]
opt_load = ["atomic ", "volatile "]
opt_addsubmul = ["nsw ", "nuw ", "nuw nsw "]
opt_usdiv = ["", "exact "]
opt_icmp = [
    "eq ",
    "ne ",
    "ugt ",
    "uge ",
    "ult ",
    "ule ",
    "sgt ",
    "sge ",
    "slt ",
    "sle ",
]
opt_fcmp = [
    "false ",
    "oeq ",
    "ogt ",
    "oge ",
    "olt ",
    "olt ",
    "ole ",
    "one ",
    "ord ",
    "ueq ",
    "ugt ",
    "uge ",
    "ult ",
    "ule ",
    "une ",
    "uno ",
    "true ",
]
opt_define = [
    "",
    "linkonce_odr ",
    "linkonce_odr ",
    "zeroext ",
    "dereferenceable\(\d+\) ",
    "hidden ",
    "internal ",
    "nonnull ",
    "weak_odr ",
    "fastcc ",
    "noalias ",
    "signext ",
    "spir_kernel ",
]
opt_invoke = [
    "",
    "dereferenceable\(\d+\) ",
    "noalias ",
    "fast ",
    "zeroext ",
    "signext ",
    "fastcc ",
]
opt_GEP = ["", "inbounds "]


# Helper functions
def any_of(possibilities, to_add=""):
    """
    Construct a regex representing "any of" the given possibilities
    :param possibilities: list of strings representing different word possibilities
    :param to_add: string to add at the beginning of each possibility (optional)
    :return: string corresponding to regex which represents any of the given possibilities
    """
    assert len(possibilities) > 0
    s = "("
    if len(to_add) > 0:
        s += possibilities[0] + to_add + " "
    else:
        s += possibilities[0]
    for i in range(len(possibilities) - 1):
        if len(to_add) > 0:
            s += "|" + possibilities[i + 1] + to_add + " "
        else:
            s += "|" + possibilities[i + 1]
    return s + ")"


# Main tags
llvm_IR_stmt_tags = [
    # ['regex'                                                    'tag'                   'tag general'
    [
        "<@ID> = (?!.*constant)(?!.*alias).*",
        "global definition",
        "global variable definition",
    ],
    ["<@ID> = .*constant .*", "global const. def.", "global variable definition"],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = add " + any_of(opt_addsubmul) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = sub " + any_of(opt_addsubmul) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = mul " + any_of(opt_addsubmul) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i16> .*",
        "<d x i16>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i32> .*",
        "<d x i32>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i64> .*",
        "<d x i64>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = udiv " + any_of(opt_usdiv) + "?<\d+ x i128> .*",
        "<d x i128>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i16> .*",
        "<d x i16>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i32> .*",
        "<d x i32>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i64> .*",
        "<d x i64>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = sdiv " + any_of(opt_usdiv) + "?<\d+ x i128> .*",
        "<d x i128>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<%ID> .*",
        "struct  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<%ID>\* .*",
        "struct*  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<%ID>\*\* .*",
        "struct**  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<%ID>\*\*\* .*",
        "struct***  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i24 .*",
        "i24 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i24> .*",
        "<d x i24> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i40 .*",
        "i40 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i40> .*",
        "<d x i40> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i1\* .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i2\* .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i4\* .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i8\* .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i16\* .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i32\* .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i40\* .*",
        "i40* operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i64\* .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i128\* .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?x86_fp80\* .*",
        "float* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?float\* .*",
        "float* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i1\*\* .*",
        "i1**  operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i2\*\* .*",
        "i2**  operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i4\*\* .*",
        "i4**  operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i8\*\* .*",
        "i8**  operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i16\*\* .*",
        "i16** operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i32\*\* .*",
        "i32** operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i40\*\* .*",
        "i40** operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i64\*\* .*",
        "i64** operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?i128\*\* .*",
        "i128** operation",
        "int** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?x86_fp80\*\* .*",
        "float** operation",
        "floating point** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?float\*\* .*",
        "float** operation",
        "floating point** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?double\*\* .*",
        "double** operation",
        "floating point** operation",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<%ID>\* .*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + '?(%"|opaque).*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = icmp " + any_of(opt_icmp) + "?void \(.*",
        "function op",
        "struct/class op",
    ],
    ["<%ID> = srem i1 .*", "i1  operation", "int operation"],
    ["<%ID> = srem <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    ["<%ID> = srem i2 .*", "i2  operation", "int operation"],
    ["<%ID> = srem <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    ["<%ID> = srem i4 .*", "i4  operation", "int operation"],
    ["<%ID> = srem <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    ["<%ID> = srem i8 .*", "i8  operation", "int operation"],
    ["<%ID> = srem <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    ["<%ID> = srem i16 .*", "i16 operation", "int operation"],
    [
        "<%ID> = srem <\d+ x i16> .*",
        "<d x i16>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = srem i32 .*", "i32 operation", "int operation"],
    [
        "<%ID> = srem <\d+ x i32> .*",
        "<d x i32>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = srem i64 .*", "i64 operation", "int operation"],
    [
        "<%ID> = srem <\d+ x i64> .*",
        "<d x i64>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = srem i128 .*", "i128 operation", "int operation"],
    [
        "<%ID> = srem <\d+ x i128> .*",
        "<d x i128>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = urem i1 .*", "i1  operation", "int operation"],
    ["<%ID> = urem <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    ["<%ID> = urem i2 .*", "i2  operation", "int operation"],
    ["<%ID> = urem <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    ["<%ID> = urem i4 .*", "i4  operation", "int operation"],
    ["<%ID> = urem <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    ["<%ID> = urem i8 .*", "i8  operation", "int operation"],
    ["<%ID> = urem <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    ["<%ID> = urem i16 .*", "i16 operation", "int operation"],
    [
        "<%ID> = urem <\d+ x i16> .*",
        "<d x i16>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = urem i32 .*", "i32 operation", "int operation"],
    [
        "<%ID> = urem <\d+ x i32> .*",
        "<d x i32>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = urem i64 .*", "i32 operation", "int operation"],
    [
        "<%ID> = urem <\d+ x i64> .*",
        "<d x i64>  operation",
        "<d x int> operation",
    ],
    ["<%ID> = urem i128 .*", "i128 operation", "int operation"],
    [
        "<%ID> = urem <\d+ x i128> .*",
        "<d x i128>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = fadd " + any_of(fast_math_flag) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = fsub " + any_of(fast_math_flag) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = fmul " + any_of(fast_math_flag) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = fdiv " + any_of(fast_math_flag) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = frem " + any_of(fast_math_flag) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?x86_fp80.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?<\d+ x x86_fp80>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?float.*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?<\d+ x float>.*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?double.*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = fcmp (fast |)?" + any_of(opt_fcmp) + "?<\d+ x double>.*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    ["<%ID> = atomicrmw add i1\* .*", "i1* operation", "int* operation"],
    ["<%ID> = atomicrmw add i2\* .*", "i2* operation", "int* operation"],
    ["<%ID> = atomicrmw add i4\* .*", "i4* operation", "int* operation"],
    ["<%ID> = atomicrmw add i8\* .*", "i8* operation", "int* operation"],
    ["<%ID> = atomicrmw add i16\* .*", "i16* operation", "int* operation"],
    ["<%ID> = atomicrmw add i32\* .*", "i32* operation", "int* operation"],
    ["<%ID> = atomicrmw add i64\* .*", "i64* operation", "int* operation"],
    ["<%ID> = atomicrmw add i128\* .*", "i128* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i1\* .*", "i1* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i2\* .*", "i2* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i4\* .*", "i4* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i8\* .*", "i8* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i16\* .*", "i16* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i32\* .*", "i32* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i64\* .*", "i64* operation", "int* operation"],
    ["<%ID> = atomicrmw sub i128\* .*", "i128* operation", "int* operation"],
    ["<%ID> = atomicrmw or i1\* .*", "i1* operation", "int* operation"],
    ["<%ID> = atomicrmw or i2\* .*", "i2* operation", "int* operation"],
    ["<%ID> = atomicrmw or i4\* .*", "i4* operation", "int* operation"],
    ["<%ID> = atomicrmw or i8\* .*", "i8* operation", "int* operation"],
    ["<%ID> = atomicrmw or i16\* .*", "i16* operation", "int* operation"],
    ["<%ID> = atomicrmw or i32\* .*", "i32* operation", "int* operation"],
    ["<%ID> = atomicrmw or i64\* .*", "i64* operation", "int* operation"],
    ["<%ID> = atomicrmw or i128\* .*", "i128* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i1\* .*", "i1* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i2\* .*", "i2* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i4\* .*", "i4* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i8\* .*", "i8* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i16\* .*", "i16* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i32\* .*", "i32* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i64\* .*", "i64* operation", "int* operation"],
    ["<%ID> = atomicrmw xchg i128\* .*", "i128* operation", "int* operation"],
    ["<%ID> = alloca i1($|,).*", "i1  operation", "int operation"],
    ["<%ID> = alloca i2($|,).*", "i2  operation", "int operation"],
    ["<%ID> = alloca i4($|,).*", "i4  operation", "int operation"],
    ["<%ID> = alloca i8($|,).*", "i8  operation", "int operation"],
    ["<%ID> = alloca i16($|,).*", "i16 operation", "int operation"],
    ["<%ID> = alloca i32($|,).*", "i32 operation", "int operation"],
    ["<%ID> = alloca i64($|,).*", "i64 operation", "int operation"],
    ["<%ID> = alloca i128($|,).*", "i128 operation", "int operation"],
    ["<%ID> = alloca i1\*($|,).*", "i1*  operation", "int* operation"],
    ["<%ID> = alloca i2\*($|,).*", "i2*  operation", "int* operation"],
    ["<%ID> = alloca i4\*($|,).*", "i4*  operation", "int* operation"],
    ["<%ID> = alloca i8\*($|,).*", "i8*  operation", "int* operation"],
    ["<%ID> = alloca i16\*($|,).*", "i16* operation", "int* operation"],
    ["<%ID> = alloca i32\*($|,).*", "i32* operation", "int* operation"],
    ["<%ID> = alloca i64\*($|,).*", "i64* operation", "int* operation"],
    ["<%ID> = alloca i128\*($|,).*", "i128* operation", "int* operation"],
    [
        "<%ID> = alloca x86_fp80($|,).*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = alloca float($|,).*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = alloca double($|,).*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = alloca x86_fp80\*($|,).*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = alloca float\*($|,).*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = alloca double\*($|,).*",
        "double* operation",
        "floating point* operation",
    ],
    ['<%ID> = alloca %".*', "struct/class op", "struct/class op"],
    ["<%ID> = alloca <%.*", "struct/class op", "struct/class op"],
    ["<%ID> = alloca <?{.*", "struct/class op", "struct/class op"],
    ["<%ID> = alloca opaque.*", "struct/class op", "struct/class op"],
    [
        "<%ID> = alloca <\d+ x i1>, .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i2>, .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i4>, .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i8>, .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i16>, .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i32>, .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i64>, .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i128>, .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = alloca <\d+ x x86_fp80>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = alloca <\d+ x float>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = alloca <\d+ x double>, .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = alloca <\d+ x \{ .* \}>, .*",
        "<d x structure> operation",
        "<d x structure> operation",
    ],
    [
        "<%ID> = alloca <\d+ x i1>\*, .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i2>\*, .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i4>\*, .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i8>\*, .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i16>\*, .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i32>\*, .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i64>\*, .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x i128>\*, .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x x86_fp80>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x float>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x double>\*, .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = alloca <\d+ x \{ .* \}>\*, .*",
        "<d x structure>* operation",
        "<d x structure>* operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i1\], .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i2\], .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i4\], .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i8\], .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i16\], .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i32\], .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i64\], .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x i128\], .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x x86_fp80\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x float\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x double\], .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = alloca \[\d+ x \{ .* \}\], .*",
        "[d x structure] operation",
        "[d x structure] operation",
    ],
    [
        "<%ID> = alloca { { float, float } }, .*",
        "{ float, float } operation",
        "complex operation",
    ],
    [
        "<%ID> = alloca { { double, double } }, .*",
        "{ double, double } operation",
        "complex operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i1, .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i2, .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i4, .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i8, .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i16, .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i24, .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i32, .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i40, .*",
        "i40 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i64, .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i128, .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i256, .*",
        "i256 operation",
        "int operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i1\*, .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i2\*, .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i4\*, .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i8\*, .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i16\*, .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i24\*, .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i32\*, .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i40\*, .*",
        "i40* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i64\*, .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i128\*, .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i256\*, .*",
        "i256* operation",
        "int* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i1\*\*, .*",
        "i1**  operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i2\*\*, .*",
        "i2**  operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i4\*\*, .*",
        "i4**  operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i8\*\*, .*",
        "i8**  operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i16\*\*, .*",
        "i16** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i24\*\*, .*",
        "i16** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i32\*\*, .*",
        "i32** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i40\*\*, .*",
        "i40** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i64\*\*, .*",
        "i64** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i128\*\*, .*",
        "i128** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i256\*\*, .*",
        "i256** operation",
        "int** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i1\*\*\*, .*",
        "i1***  operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i2\*\*\*, .*",
        "i2***  operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i4\*\*\*, .*",
        "i4***  operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i8\*\*\*, .*",
        "i8***  operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i16\*\*\*, .*",
        "i16*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i24\*\*\*, .*",
        "i16*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i32\*\*\*, .*",
        "i32*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i40\*\*\*, .*",
        "i40*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i64\*\*\*, .*",
        "i64*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i128\*\*\*, .*",
        "i128*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?i256\*\*\*, .*",
        "i256*** operation",
        "int*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?x86_fp80, .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?float, .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?double, .*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?x86_fp80\*, .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?float\*, .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?double\*, .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?x86_fp80\*\*, .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?float\*\*, .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?double\*\*, .*",
        "double** operation",
        "floating point** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?x86_fp80\*\*\*, .*",
        "float***  operation",
        "floating point*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?float\*\*\*, .*",
        "float***  operation",
        "floating point*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?double\*\*\*, .*",
        "double*** operation",
        "floating point*** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + '?%".*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<%.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i1>, .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i2>, .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i4>, .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i8>, .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i16>, .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i24>, .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i32>, .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i40>, .*",
        "<d x i40> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i64>, .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i128>, .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x x86_fp80>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x float>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x double>, .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x \{ .* \}>, .*",
        "<d x structure> operation",
        "<d x structure> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i1\*>, .*",
        "<d x i1*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i2\*>, .*",
        "<d x i2*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i4\*>, .*",
        "<d x i4*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i8\*>, .*",
        "<d x i8*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i16\*>, .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i24\*>, .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i32\*>, .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i40\*>, .*",
        "<d x i40*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i64\*>, .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i128\*>, .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x x86_fp80\*>, .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x float\*>, .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x double\*>, .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i1>\*, .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i2>\*, .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i4>\*, .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i8>\*, .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i16>\*, .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i24>\*, .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i32>\*, .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i40>\*, .*",
        "<d x i40>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i64>\*, .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x i128>\*, .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x x86_fp80>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x float>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x double>\*, .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x \{ .* \}>\*, .*",
        "<d x structure>* operation",
        "<d x structure>* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x x86_fp80>\*\*, .*",
        "<d x float>** operation",
        "<d x floating point>** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x float>\*\*, .*",
        "<d x float>** operation",
        "<d x floating point>** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x double>\*\*, .*",
        "<d x double>** operation",
        "<d x floating point>** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?<\d+ x \{ .* \}>\*\*, .*",
        "<d x structure>** operation",
        "<d x structure>** operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i1\], .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i2\], .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i4\], .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i8\], .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i16\], .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i24\], .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i32\], .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i40\], .*",
        "[d x i40] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i64\], .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i128\], .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x x86_fp80\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x float\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x double\], .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x \{ .* \}\], .*",
        "[d x structure] operation",
        "[d x structure] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i1\]\*, .*",
        "[d x i1]*  operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i2\]\*, .*",
        "[d x i2]*  operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i4\]\*, .*",
        "[d x i4]*  operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i8\]\*, .*",
        "[d x i8]*  operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i16\]\*, .*",
        "[d x i16]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i32\]\*, .*",
        "[d x i32]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i40\]\*, .*",
        "[d x i40]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i64\]\*, .*",
        "[d x i64]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x i128\]\*, .*",
        "[d x i128]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x x86_fp80\]\*, .*",
        "[d x float]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x float\]\*, .*",
        "[d x float]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x double\]\*, .*",
        "[d x double]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?\[\d+ x \{ .* \}\]\*, .*",
        "[d x structure]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = load " + any_of(opt_load) + "?.*\(.*\)\*+, .*",
        "function operation",
        "function operation",
    ],
    ["store " + any_of(opt_load) + "?i1 .*", "i1  operation", "int operation"],
    ["store " + any_of(opt_load) + "?i2 .*", "i2  operation", "int operation"],
    ["store " + any_of(opt_load) + "?i4 .*", "i4  operation", "int operation"],
    ["store " + any_of(opt_load) + "?i8 .*", "i8  operation", "int operation"],
    ["store " + any_of(opt_load) + "?i16 .*", "i16 operation", "int operation"],
    ["store " + any_of(opt_load) + "?i24 .*", "i16 operation", "int operation"],
    ["store " + any_of(opt_load) + "?i32 .*", "i32 operation", "int operation"],
    ["store " + any_of(opt_load) + "?i40 .*", "i32 operation", "int operation"],
    ["store " + any_of(opt_load) + "?i64 .*", "i64 operation", "int operation"],
    ["store " + any_of(opt_load) + "?i128 .*", "i128 operation", "int operation"],
    [
        "store " + any_of(opt_load) + "?i1\* .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i2\* .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i4\* .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i8\* .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i16\* .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i32\* .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i64\* .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i128\* .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "store " + any_of(opt_load) + "?i1\*\* .*",
        "i1**  operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i2\*\* .*",
        "i2**  operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i4\*\* .*",
        "i4**  operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i8\*\* .*",
        "i8**  operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i16\*\* .*",
        "i16** operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i32\*\* .*",
        "i32** operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i64\*\* .*",
        "i64** operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?i128\*\* .*",
        "i128** operation",
        "int** operation",
    ],
    [
        "store " + any_of(opt_load) + "?x86_fp80 .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "store " + any_of(opt_load) + "?float .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "store " + any_of(opt_load) + "?double .*",
        "double operation",
        "floating point operation",
    ],
    [
        "store " + any_of(opt_load) + "?x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "store " + any_of(opt_load) + "?float\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "store " + any_of(opt_load) + "?double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "store " + any_of(opt_load) + "?x86_fp80\*\* .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "store " + any_of(opt_load) + "?float\*\* .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "store " + any_of(opt_load) + "?double\*\* .*",
        "double** operation",
        "floating point** operation",
    ],
    ["store " + any_of(opt_load) + "?void \(.*", "function op", "function op"],
    ["store " + any_of(opt_load) + '?%".*', "struct/class op", "struct/class op"],
    ["store " + any_of(opt_load) + "?<%.*", "struct/class op", "struct/class op"],
    [
        "store " + any_of(opt_load) + "?<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "store " + any_of(opt_load) + "?opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x \{ .* \}> .*",
        "<d x \{ .* \}> operation",
        "<d x \{ .* \}> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i1\*> .*",
        "<d x i1*>  operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i2\*> .*",
        "<d x i2*>  operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i4\*> .*",
        "<d x i4*>  operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i8\*> .*",
        "<d x i8*>  operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i16\*> .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i32\*> .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i64\*> .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i128\*> .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x x86_fp80\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x float\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x double\*> .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x \{ .* \}\*> .*",
        "<d x \{ .* \}*> operation",
        "<d x \{ .* \}*> operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i1>\* .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i2>\* .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i4>\* .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i8>\* .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i16>\* .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i32>\* .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i64>\* .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x i128>\* .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x x86_fp80>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x \{ .* \}\*?>\* .*",
        "<d x struct>* operation",
        "<d x \{ .* \}>* operation",
    ],
    [
        "store " + any_of(opt_load) + "?<\d+ x void \(.*",
        "<d x function>* operation",
        "<d x function operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i1\] .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i2\] .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i4\] .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i8\] .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i16\] .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i32\] .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i64\] .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x i128\] .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x x86_fp80\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x float\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x double\] .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "store " + any_of(opt_load) + "?\[\d+ x \{ .* \}\] .*",
        "[d x structure] operation",
        "[d x structure] operation",
    ],
    ["declare (noalias |nonnull )*void .*", "void operation", "void operation"],
    ["declare (noalias |nonnull )*i1 .*", "i1  operation", "int operation"],
    ["declare (noalias |nonnull )*i2 .*", "i2  operation", "int operation"],
    ["declare (noalias |nonnull )*i4 .*", "i4  operation", "int operation"],
    ["declare (noalias |nonnull )*i8 .*", "i8  operation", "int operation"],
    ["declare (noalias |nonnull )*i16 .*", "i16 operation", "int operation"],
    ["declare (noalias |nonnull )*i32 .*", "i32 operation", "int operation"],
    ["declare (noalias |nonnull )*i64 .*", "i64 operation", "int operation"],
    ["declare (noalias |nonnull )*i8\* .*", "i8*  operation", "int* operation"],
    ["declare (noalias |nonnull )*i16\* .*", "i16* operation", "int* operation"],
    ["declare (noalias |nonnull )*i32\* .*", "i32* operation", "int* operation"],
    ["declare (noalias |nonnull )*i64\* .*", "i64* operation", "int* operation"],
    [
        "declare (noalias |nonnull )*x86_fp80 .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "declare (noalias |nonnull )*float .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "declare (noalias |nonnull )*double .*",
        "double operation",
        "floating point operation",
    ],
    [
        "declare (noalias |nonnull )*x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "declare (noalias |nonnull )*float\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "declare (noalias |nonnull )*double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    ['declare (noalias |nonnull )*%".*', "struct/class op", "struct/class op"],
    ["declare (noalias |nonnull )*<%.*", "struct/class op", "struct/class op"],
    ["declare (noalias |nonnull )*<?{.*", "struct/class op", "struct/class op"],
    [
        "declare (noalias |nonnull )*opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i1>\* .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i2>\* .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i4>\* .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i8>\* .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i16>\* .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i32>\* .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i64>\* .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x i128>\* .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x x86_fp80>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "declare (noalias |nonnull )*<\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i1\] .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i2\] .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i4\] .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i8\] .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i16\] .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i32\] .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i64\] .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x i128\] .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x x86_fp80\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x float\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "declare (noalias |nonnull )*\[\d+ x double\] .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "define " + any_of(opt_define) + "+void .*",
        "void operation",
        "void operation",
    ],
    ["define " + any_of(opt_define) + "+i1 .*", "i1  operation", "int operation"],
    ["define " + any_of(opt_define) + "+i2 .*", "i2  operation", "int operation"],
    ["define " + any_of(opt_define) + "+i4 .*", "i4  operation", "int operation"],
    ["define " + any_of(opt_define) + "+i8 .*", "i8  operation", "int operation"],
    [
        "define " + any_of(opt_define) + "+i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "define " + any_of(opt_define) + "+i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "define " + any_of(opt_define) + "+i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "define " + any_of(opt_define) + "+i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "define " + any_of(opt_define) + "+i1\* .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i2\* .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i4\* .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i8\* .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i16\* .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i32\* .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i64\* .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+i128\* .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "define " + any_of(opt_define) + "+x86_fp80 .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "define " + any_of(opt_define) + "+float .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "define " + any_of(opt_define) + "+double .*",
        "double operation",
        "floating point operation",
    ],
    [
        "define " + any_of(opt_define) + "+x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "define " + any_of(opt_define) + "+float\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "define " + any_of(opt_define) + "+double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "define " + any_of(opt_define) + '+%".*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "define " + any_of(opt_define) + "+<%.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "define " + any_of(opt_define) + "+<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "define " + any_of(opt_define) + "+opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i1>\* .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i2>\* .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i4>\* .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i8>\* .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i16>\* .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i32>\* .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i64>\* .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x i128>\* .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x x86_fp80>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+<\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i1\] .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i2\] .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i4\] .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i8\] .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i16\] .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i32\] .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i64\] .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x i128\] .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x x86_fp80\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x float\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "define " + any_of(opt_define) + "+\[\d+ x double\] .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i1\* .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i2\* .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i4\* .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i8\* .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i16\* .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i32\* .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i64\* .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i128\* .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i1\*\* .*",
        "i1**  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i2\*\* .*",
        "i2**  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i4\*\* .*",
        "i4**  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i8\*\* .*",
        "i8**  operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i16\*\* .*",
        "i16** operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i32\*\* .*",
        "i32** operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i64\*\* .*",
        "i64** operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*i128\*\* .*",
        "i128** operation",
        "int* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*x86_fp80 .*",
        "float operation",
        "floating point operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*float .*",
        "float operation",
        "floating point operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*double .*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*x86_fp80\* .*",
        "float* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*float\* .*",
        "float* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*x86_fp80\*\* .*",
        "float** operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*float\*\* .*",
        "float** operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*double\*\* .*",
        "double** operation",
        "floating point* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + '*%".*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*<%.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call " + any_of(opt_invoke) + "*opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i1>\* .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i2>\* .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i4>\* .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i8>\* .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i16>\* .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i32>\* .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i64>\* .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x i128>\* .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x x86_fp80>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*<\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i1\] .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i2\] .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i4\] .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i8\] .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i16\] .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i32\] .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i64\] .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x i128\] .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x x86_fp80\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x float\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = (tail |musttail |notail )?call "
        + any_of(opt_invoke)
        + "*\[\d+ x double\] .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    ["ret i1 .*", "i1  operation", "int operation"],
    ["ret i2 .*", "i2  operation", "int operation"],
    ["ret i4 .*", "i4  operation", "int operation"],
    ["ret i8 .*", "i8  operation", "int operation"],
    ["ret i16 .*", "i16 operation", "int operation"],
    ["ret i32 .*", "i32 operation", "int operation"],
    ["ret i64 .*", "i64 operation", "int operation"],
    ["ret i128 .*", "i128 operation", "int operation"],
    ["ret i1\* .*", "i1*  operation", "int* operation"],
    ["ret i2\* .*", "i2*  operation", "int* operation"],
    ["ret i4\* .*", "i4*  operation", "int* operation"],
    ["ret i8\* .*", "i8*  operation", "int* operation"],
    ["ret i16\* .*", "i16* operation", "int* operation"],
    ["ret i32\* .*", "i32* operation", "int* operation"],
    ["ret i64\* .*", "i64* operation", "int* operation"],
    ["ret i128\* .*", "i128* operation", "int* operation"],
    ["ret x86_fp80 .*", "x86_fp80  operation", "floating point operation"],
    ["ret float .*", "float  operation", "floating point operation"],
    ["ret double .*", "double operation", "floating point operation"],
    ["ret x86_fp80\* .*", "x86_fp80*  operation", "floating point* operation"],
    ["ret float\* .*", "float*  operation", "floating point* operation"],
    ["ret double\* .*", "double* operation", "floating point* operation"],
    ['ret %".*', "struct/class op", "struct/class op"],
    ["ret <%.*", "struct/class op", "struct/class op"],
    ["ret <?{.*", "struct/class op", "struct/class op"],
    ["ret opaque.*", "struct/class op", "struct/class op"],
    ["ret <\d+ x i1> .*", "<d x i1>  operation", "<d x int> operation"],
    ["ret <\d+ x i2> .*", "<d x i2>  operation", "<d x int> operation"],
    ["ret <\d+ x i4> .*", "<d x i4>  operation", "<d x int> operation"],
    ["ret <\d+ x i8> .*", "<d x i8>  operation", "<d x int> operation"],
    ["ret <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    ["ret <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    ["ret <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    ["ret <\d+ x i128> .*", "<d x i128> operation", "<d x int> operation"],
    [
        "ret <\d+ x x86_fp80> .*",
        "<d x x86_fp80> operation",
        "<d x floating point> operation",
    ],
    [
        "ret <\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "ret <\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    ["ret <\d+ x i1>\* .*", "<d x i1>*  operation", "<d x int>* operation"],
    ["ret <\d+ x i2>\* .*", "<d x i2>*  operation", "<d x int>* operation"],
    ["ret <\d+ x i4>\* .*", "<d x i4>*  operation", "<d x int>* operation"],
    ["ret <\d+ x i8>\* .*", "<d x i8>*  operation", "<d x int>* operation"],
    ["ret <\d+ x i16>\* .*", "<d x i16>* operation", "<d x int>* operation"],
    ["ret <\d+ x i32>\* .*", "<d x i32>* operation", "<d x int>* operation"],
    ["ret <\d+ x i64>\* .*", "<d x i64>* operation", "<d x int>* operation"],
    ["ret <\d+ x i128>\* .*", "<d x i128>* operation", "<d x int>* operation"],
    [
        "ret <\d+ x x86_fp80>\* .*",
        "<d x x86_fp80>* operation",
        "<d x floating point>* operation",
    ],
    [
        "ret <\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "ret <\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    ["ret \[\d+ x i1\] .*", "[d x i1]  operation", "[d x int] operation"],
    ["ret \[\d+ x i2\] .*", "[d x i2]  operation", "[d x int] operation"],
    ["ret \[\d+ x i4\] .*", "[d x i4]  operation", "[d x int] operation"],
    ["ret \[\d+ x i8\] .*", "[d x i8]  operation", "[d x int] operation"],
    ["ret \[\d+ x i16\] .*", "[d x i16] operation", "[d x int] operation"],
    ["ret \[\d+ x i32\] .*", "[d x i32] operation", "[d x int] operation"],
    ["ret \[\d+ x i64\] .*", "[d x i64] operation", "[d x int] operation"],
    ["ret \[\d+ x i128\] .*", "[d x i128] operation", "[d x int] operation"],
    [
        "ret \[\d+ x x86_fp80\] .*",
        "[d x x86_fp80] operation",
        "[d x floating point] operation",
    ],
    [
        "ret \[\d+ x float\] .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "ret \[\d+ x double\] .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    ["<%ID> = and i1 .*", "i1  operation", "int operation"],
    ["<%ID> = and <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    ["<%ID> = and i2 .*", "i2  operation", "int operation"],
    ["<%ID> = and <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    ["<%ID> = and i4 .*", "i4  operation", "int operation"],
    ["<%ID> = and <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    ["<%ID> = and i8 .*", "i8  operation", "int operation"],
    ["<%ID> = and <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    ["<%ID> = and i16 .*", "i16 operation", "int operation"],
    ["<%ID> = and <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    ["<%ID> = and i24 .*", "i24 operation", "int operation"],
    ["<%ID> = and <\d+ x i24> .*", "<d x i24> operation", "<d x int> operation"],
    ["<%ID> = and i32 .*", "i32 operation", "int operation"],
    ["<%ID> = and <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    ["<%ID> = and i40 .*", "i40 operation", "int operation"],
    ["<%ID> = and <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    ["<%ID> = and i64 .*", "i64 operation", "int operation"],
    ["<%ID> = and <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    ["<%ID> = and i128 .*", "i128 operation", "int operation"],
    [
        "<%ID> = and <\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    ["<%ID> = or i1 .*", "i1  operation", "int operation"],
    ["<%ID> = or <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    ["<%ID> = or i2 .*", "i2  operation", "int operation"],
    ["<%ID> = or <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    ["<%ID> = or i4 .*", "i4  operation", "int operation"],
    ["<%ID> = or <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    ["<%ID> = or i8 .*", "i8  operation", "int operation"],
    ["<%ID> = or <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    ["<%ID> = or i16 .*", "i16 operation", "int operation"],
    ["<%ID> = or <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    ["<%ID> = or i24 .*", "i24 operation", "int operation"],
    ["<%ID> = or <\d+ x i24> .*", "<d x i24> operation", "<d x int> operation"],
    ["<%ID> = or i32 .*", "i32 operation", "int operation"],
    ["<%ID> = or <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    ["<%ID> = or i40 .*", "i40 operation", "int operation"],
    ["<%ID> = or <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    ["<%ID> = or i64 .*", "i64 operation", "int operation"],
    ["<%ID> = or <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    ["<%ID> = or i128 .*", "i128 operation", "int operation"],
    ["<%ID> = or <\d+ x i128> .*", "<d x i128> operation", "<d x int> operation"],
    ["<%ID> = xor i1 .*", "i1  operation", "int operation"],
    ["<%ID> = xor <\d+ x i1>.*", "<d x i1> operation", "<d x int> operation"],
    ["<%ID> = xor i4 .*", "i4  operation", "int operation"],
    ["<%ID> = xor <\d+ x i2>.*", "<d x i2> operation", "<d x int> operation"],
    ["<%ID> = xor i2 .*", "i2  operation", "int operation"],
    ["<%ID> = xor <\d+ x i4>.*", "<d x i4> operation", "<d x int> operation"],
    ["<%ID> = xor i8 .*", "i8  operation", "int operation"],
    ["<%ID> = xor <\d+ x i8>.*", "<d x i8> operation", "<d x int> operation"],
    ["<%ID> = xor i16 .*", "i16 operation", "int operation"],
    ["<%ID> = xor <\d+ x i16>.*", "<d x i16> operation", "<d x int> operation"],
    ["<%ID> = xor i24 .*", "i16 operation", "int operation"],
    ["<%ID> = xor <\d+ x i24>.*", "<d x i16> operation", "<d x int> operation"],
    ["<%ID> = xor i32 .*", "i32 operation", "int operation"],
    ["<%ID> = xor <\d+ x i32>.*", "<d x i32> operation", "<d x int> operation"],
    ["<%ID> = xor i40 .*", "i40 operation", "int operation"],
    ["<%ID> = xor <\d+ x i40>.*", "<d x i40> operation", "<d x int> operation"],
    ["<%ID> = xor i64 .*", "i64 operation", "int operation"],
    ["<%ID> = xor <\d+ x i64>.*", "<d x i64> operation", "<d x int> operation"],
    ["<%ID> = xor i128 .*", "i128 operation", "int operation"],
    ["<%ID> = xor <\d+ x i128>.*", "<d x i128> operation", "<d x int> operation"],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i4 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i40 .*",
        "i40 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i40> .*",
        "<d x i40> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?i256 .*",
        "i256 operation",
        "int operation",
    ],
    [
        "<%ID> = shl " + any_of(opt_addsubmul) + "?<\d+ x i256> .*",
        "<d x i256> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i1 .*",
        "i1 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i2 .*",
        "i2 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i40 .*",
        "i40 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i40> .*",
        "<d x i40> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?i256 .*",
        "i256 operation",
        "int operation",
    ],
    [
        "<%ID> = ashr " + any_of(opt_usdiv) + "?<\d+ x i256> .*",
        "<d x i256> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i1> .*",
        "<d x i1> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i2> .*",
        "<d x i2> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i4> .*",
        "<d x i4> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i8> .*",
        "<d x i8> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i24 .*",
        "i24 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i24> .*",
        "<d x i24> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i40 .*",
        "i40 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i40> .*",
        "<d x i40> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?i256 .*",
        "i256 operation",
        "int operation",
    ],
    [
        "<%ID> = lshr " + any_of(opt_usdiv) + "?<\d+ x i256> .*",
        "<d x i256> operation",
        "<d x int> operation",
    ],
    ["<%ID> = phi i1 .*", "i1  operation", "int operation"],
    ["<%ID> = phi <\d+ x i1> .*", "<d x i1> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i1\*> .*",
        "<d x i1*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i1>\* .*",
        "<d x i1>* operation",
        "<d x int>* operation",
    ],
    ["<%ID> = phi \[\d+ x i1\] .*", "[d x i1] operation", "[d x int] operation"],
    [
        "<%ID> = phi \[\d+ x i1\]\* .*",
        "[d x i1]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i1\]\*\* .*",
        "[d x i1]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i1\]\*\*\* .*",
        "[d x i1]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i2 .*", "i2  operation", "int operation"],
    ["<%ID> = phi <\d+ x i2> .*", "<d x i2> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i2\*> .*",
        "<d x i2*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i2>\* .*",
        "<d x i2>* operation",
        "<d x int>* operation",
    ],
    ["<%ID> = phi \[\d+ x i2\] .*", "[d x i2] operation", "[d x int] operation"],
    [
        "<%ID> = phi \[\d+ x i2\]\* .*",
        "[d x i2]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i2\]\*\* .*",
        "[d x i2]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i2\]\*\*\* .*",
        "[d x i2]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i4 .*", "i4  operation", "int operation"],
    ["<%ID> = phi <\d+ x i4> .*", "<d x i4> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i4\*> .*",
        "<d x i4*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i4>\* .*",
        "<d x i4>* operation",
        "<d x int>* operation",
    ],
    ["<%ID> = phi \[\d+ x i4\] .*", "[d x i4] operation", "[d x int] operation"],
    [
        "<%ID> = phi \[\d+ x i4\]\* .*",
        "[d x i4]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i4\]\*\* .*",
        "[d x i4]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i4\]\*\*\* .*",
        "[d x i4]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i8 .*", "i8  operation", "int operation"],
    ["<%ID> = phi <\d+ x i8> .*", "<d x i8> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i8\*> .*",
        "<d x i8*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i8>\* .*",
        "<d x i8>* operation",
        "<d x int>* operation",
    ],
    ["<%ID> = phi \[\d+ x i8\] .*", "[d x i4] operation", "[d x int] operation"],
    [
        "<%ID> = phi \[\d+ x i8\]\* .*",
        "[d x i4]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i8\]\*\* .*",
        "[d x i4]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i8\]\*\*\* .*",
        "[d x i4]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i16 .*", "i16 operation", "int operation"],
    ["<%ID> = phi <\d+ x i16> .*", "<d x i16> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i16\*> .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i16>\* .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i16\] .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = phi \[\d+ x i16\]\* .*",
        "[d x i16]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i16\]\*\* .*",
        "[d x i16]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i16\]\*\*\* .*",
        "[d x i16]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i32 .*", "i32 operation", "int operation"],
    ["<%ID> = phi <\d+ x i32> .*", "<d x i32> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i32\*> .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i32>\* .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i32\] .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = phi \[\d+ x i32\]\* .*",
        "[d x i32]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i32\]\*\* .*",
        "[d x i32]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i32\]\*\*\* .*",
        "[d x i32]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i40 .*", "i32 operation", "int operation"],
    ["<%ID> = phi <\d+ x i40> .*", "<d x i40> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i40\*> .*",
        "<d x i40*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i40>\* .*",
        "<d x i40>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i40\] .*",
        "[d x i40] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = phi \[\d+ x i40\]\* .*",
        "[d x i40]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i40\]\*\* .*",
        "[d x i40]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i40\]\*\*\* .*",
        "[d x i40]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i64 .*", "i64 operation", "int operation"],
    ["<%ID> = phi <\d+ x i64> .*", "<d x i64> operation", "<d x int> operation"],
    [
        "<%ID> = phi <\d+ x i64\*> .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i64>\* .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i64\] .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = phi \[\d+ x i64\]\* .*",
        "[d x i64]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i64\]\*\* .*",
        "[d x i64]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i64\]\*\*\* .*",
        "[d x i64]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i128 .*", "i128 operation", "int operation"],
    [
        "<%ID> = phi <\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = phi <\d+ x i128\*> .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = phi <\d+ x i128>\* .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i128\] .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = phi \[\d+ x i128\]\* .*",
        "[d x i128]* operation",
        "[d x int]* operation",
    ],
    [
        "<%ID> = phi \[\d+ x i126\]\*\* .*",
        "[d x i128]** operation",
        "[d x int]** operation",
    ],
    [
        "<%ID> = phi \[\d+ x i128\]\*\*\* .*",
        "[d x i128]*** operation",
        "[d x int]*** operation",
    ],
    ["<%ID> = phi i1\* .*", "i1*  operation", "int* operation"],
    ["<%ID> = phi i2\* .*", "i2*  operation", "int* operation"],
    ["<%ID> = phi i4\* .*", "i4*  operation", "int* operation"],
    ["<%ID> = phi i8\* .*", "i8*  operation", "int* operation"],
    ["<%ID> = phi i16\* .*", "i16*  operation", "int* operation"],
    ["<%ID> = phi i32\* .*", "i32*  operation", "int* operation"],
    ["<%ID> = phi i40\* .*", "i40*  operation", "int* operation"],
    ["<%ID> = phi i64\* .*", "i64*  operation", "int* operation"],
    ["<%ID> = phi i128\* .*", "i128*  operation", "int* operation"],
    ["<%ID> = phi i1\*\* .*", "i1**  operation", "int** operation"],
    ["<%ID> = phi i2\*\* .*", "i2**  operation", "int** operation"],
    ["<%ID> = phi i4\*\* .*", "i4**  operation", "int** operation"],
    ["<%ID> = phi i8\*\* .*", "i8**  operation", "int** operation"],
    ["<%ID> = phi i16\*\* .*", "i16** operation", "int** operation"],
    ["<%ID> = phi i32\*\* .*", "i32** operation", "int** operation"],
    ["<%ID> = phi i40\*\* .*", "i40** operation", "int** operation"],
    ["<%ID> = phi i64\*\* .*", "i64** operation", "int** operation"],
    ["<%ID> = phi i128\*\* .*", "i128** operation", "int** operation"],
    ["<%ID> = phi i1\*\*\* .*", "i1***  operation", "int*** operation"],
    ["<%ID> = phi i2\*\*\* .*", "i2***  operation", "int*** operation"],
    ["<%ID> = phi i4\*\*\* .*", "i4***  operation", "int*** operation"],
    ["<%ID> = phi i8\*\*\* .*", "i8***  operation", "int*** operation"],
    ["<%ID> = phi i16\*\*\* .*", "i16*** operation", "int*** operation"],
    ["<%ID> = phi i32\*\*\* .*", "i32*** operation", "int*** operation"],
    ["<%ID> = phi i64\*\*\* .*", "i64*** operation", "int*** operation"],
    ["<%ID> = phi i128\*\*\* .*", "i128*** operation", "int*** operation"],
    ["<%ID> = phi x86_fp80 .*", "float  operation", "floating point operation"],
    ["<%ID> = phi float .*", "float  operation", "floating point operation"],
    ["<%ID> = phi double .*", "double operation", "floating point operation"],
    [
        "<%ID> = phi <\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = phi <\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = phi <\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = phi x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = phi <\d+ x x86_fp80\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = phi <\d+ x float\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = phi <\d+ x double\*> .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = phi <\d+ x x86_fp80>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = phi <\d+ x float>\* .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = phi <\d+ x double>\* .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = phi x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    ["<%ID> = phi float\* .*", "float*  operation", "floating point* operation"],
    ["<%ID> = phi double\* .*", "double* operation", "floating point* operation"],
    [
        "<%ID> = phi x86_fp80\*\* .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = phi float\*\* .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = phi double\*\* .*",
        "double**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = phi x86_fp80\*\*\* .*",
        "float***  operation",
        "floating point*** operation",
    ],
    [
        "<%ID> = phi float\*\*\* .*",
        "float***  operation",
        "floating point*** operation",
    ],
    [
        "<%ID> = phi double\*\*\* .*",
        "double***  operation",
        "floating point*** operation",
    ],
    ["<%ID> = phi void \(.*\) \[.*", "function op", "function op"],
    ["<%ID> = phi void \(.*\)\* \[.*", "function* op", "function* op"],
    ["<%ID> = phi void \(.*\)\*\* \[.*", "function** op", "function** op"],
    ["<%ID> = phi void \(.*\)\*\*\* \[.*", "function*** op", "function*** op"],
    ["<%ID> = phi (<?{|opaque|<%ID>) .*", "struct/class op", "struct/class op"],
    [
        "<%ID> = phi (<?{|opaque|<%ID>)\* .*",
        "struct/class* op",
        "struct/class* op",
    ],
    [
        "<%ID> = phi (<?{|opaque|<%ID>)\*\* .*",
        "struct/class** op",
        "struct/class** op",
    ],
    [
        "<%ID> = phi (<?{|opaque|<%ID>)\*\*\* .*",
        "struct/class*** op",
        "struct/class*** op",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i1, .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i2, .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i4, .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i8, .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i16, .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i32, .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i64, .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i128, .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i1\*, .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i2\*, .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i4\*, .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i8\*, .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i16\*, .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i32\*, .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i64\*, .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i128\*, .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i1\*\*, .*",
        "i1**  operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i2\*\*, .*",
        "i2**  operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i4\*\*, .*",
        "i4**  operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i8\*\*, .*",
        "i8**  operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i16\*\*, .*",
        "i16** operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i32\*\*, .*",
        "i32** operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i64\*\*, .*",
        "i64** operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "i128\*\*, .*",
        "i128** operation",
        "int** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "x86_fp80, .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "float, .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "double, .*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "x86_fp80\*, .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "float\*, .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "double\*, .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "x86_fp80\*\*, .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "float\*\*, .*",
        "float**  operation",
        "floating point** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "double\*\*, .*",
        "double** operation",
        "floating point** operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + '%".*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<%.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i1>, .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i2>, .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i4>, .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i8>, .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i16>, .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i32>, .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i64>, .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i128>, .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x x86_fp80>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x float>, .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x double>, .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i1>\*, .*",
        "<d x i1>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i2>\*, .*",
        "<d x i2>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i4>\*, .*",
        "<d x i4>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i8>\*, .*",
        "<d x i8>*  operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i16>\*, .*",
        "<d x i16>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i32>\*, .*",
        "<d x i32>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i64>\*, .*",
        "<d x i64>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x i128>\*, .*",
        "<d x i128>* operation",
        "<d x int>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x x86_fp80>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x float>\*, .*",
        "<d x float>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "<\d+ x double>\*, .*",
        "<d x double>* operation",
        "<d x floating point>* operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i1\], .*",
        "[d x i1]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i2\], .*",
        "[d x i2]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i4\], .*",
        "[d x i4]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i8\], .*",
        "[d x i8]  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i16\], .*",
        "[d x i16] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i32\], .*",
        "[d x i32] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i64\], .*",
        "[d x i64] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i128\], .*",
        "[d x i128] operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x x86_fp80\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x float\], .*",
        "[d x float] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x double\], .*",
        "[d x double] operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x .*\], .*",
        "array of array operation",
        "array of array operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i1\]\*, .*",
        "[d x i1]*  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i2\]\*, .*",
        "[d x i2]*  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i4\]\*, .*",
        "[d x i4]*  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i8\]\*, .*",
        "[d x i8]*  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i16\]\*, .*",
        "[d x i16]* operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i32\]\*, .*",
        "[d x i32]* operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i64\]\*, .*",
        "[d x i64]* operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i128\]\*, .*",
        "[d x i128]* operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x x86_fp80\]\*, .*",
        "[d x float]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x float\]\*, .*",
        "[d x float]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x double\]\*, .*",
        "[d x double]* operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x .*\]\*, .*",
        "array of array* operation",
        "array of array operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i1\]\*\*, .*",
        "[d x i1]**  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i2\]\*\*, .*",
        "[d x i2]**  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i4\]\*\*, .*",
        "[d x i4]**  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i8\]\*\*, .*",
        "[d x i8]**  operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i16\]\*\*, .*",
        "[d x i16]** operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i32\]\*\*, .*",
        "[d x i32]** operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i64\]\*\*, .*",
        "[d x i64]** operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x i128\]\*\*, .*",
        "[d x i128]** operation",
        "[d x int] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x x86_fp80\]\*\*, .*",
        "[d x float]** operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x float\]\*\*, .*",
        "[d x float]** operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x double\]\*\*, .*",
        "[d x double]** operation",
        "[d x floating point] operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + "\[\d+ x .*\], .*",
        "array of array** operation",
        "array of array operation",
    ],
    [
        "<%ID> = getelementptr " + any_of(opt_GEP) + ".*\(.*\)\*+, .*",
        "function operation",
        "function operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i1 .*",
        "i1  operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i2 .*",
        "i2  operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i4 .*",
        "i4  operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i8 .*",
        "i8  operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i16 .*",
        "i16 operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i32 .*",
        "i32 operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i64 .*",
        "i64 operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i128 .*",
        "i128 operation",
        "int operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i1\* .*",
        "i1*  operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i2\* .*",
        "i2*  operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i4\* .*",
        "i4*  operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i8\* .*",
        "i8*  operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i16\* .*",
        "i16* operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i32\* .*",
        "i32* operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i64\* .*",
        "i64* operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*i128\* .*",
        "i128* operation",
        "int* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*x86_fp80 .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*float .*",
        "float  operation",
        "floating point operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*double .*",
        "double operation",
        "floating point operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*x86_fp80\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*float\* .*",
        "float*  operation",
        "floating point* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*double\* .*",
        "double* operation",
        "floating point* operation",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + '*%".*',
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*<?{.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + "*opaque.*",
        "struct/class op",
        "struct/class op",
    ],
    [
        "<%ID> = invoke " + any_of(opt_invoke) + '*%".*\*.*',
        "struct/class* op",
        "struct/class op",
    ],
    ["<%ID> = invoke " + any_of(opt_invoke) + "*void .*", "void op", "void op"],
    ["invoke " + any_of(opt_invoke) + "*void .*", "void op", "void op"],
    [
        "<%ID> = extractelement <\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i1\*> .*",
        "<d x i1*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i2\*> .*",
        "<d x i2*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i4\*> .*",
        "<d x i4*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i8\*> .*",
        "<d x i8*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i16\*> .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i32\*> .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i64\*> .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x i128\*> .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x x86_fp80\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x float\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x double\*> .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x \{.*\}> .*",
        "<d x struct> operation",
        "<d x struct> operation",
    ],
    [
        "<%ID> = extractelement <\d+ x \{.*\}\*> .*",
        "<d x struct*> operation",
        "<d x struct*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i1\*> .*",
        "<d x i1*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i2\*> .*",
        "<d x i2*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i4\*> .*",
        "<d x i4*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i8\*> .*",
        "<d x i8*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i16\*> .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i32\*> .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i64\*> .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x i128\*> .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x x86_fp80\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x float\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x double\*> .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x \{.*\}> .*",
        "<d x struct> operation",
        "<d x struct> operation",
    ],
    [
        "<%ID> = insertelement <\d+ x \{.*\}\*> .*",
        "<d x struct*> operation",
        "<d x struct*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i1> .*",
        "<d x i1>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i1\*> .*",
        "<d x i1*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i2> .*",
        "<d x i2>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i2\*> .*",
        "<d x i2*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i4> .*",
        "<d x i4>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i4\*> .*",
        "<d x i4*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i8> .*",
        "<d x i8>  operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i8\*> .*",
        "<d x i8*>  operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i16> .*",
        "<d x i16> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i16\*> .*",
        "<d x i16*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i32> .*",
        "<d x i32> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i32\*> .*",
        "<d x i32*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i64> .*",
        "<d x i64> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i64\*> .*",
        "<d x i64*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i128> .*",
        "<d x i128> operation",
        "<d x int> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x i128\*> .*",
        "<d x i128*> operation",
        "<d x int*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x x86_fp80> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x x86_fp80\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x float> .*",
        "<d x float> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x float\*> .*",
        "<d x float*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x double> .*",
        "<d x double> operation",
        "<d x floating point> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x double\*> .*",
        "<d x double*> operation",
        "<d x floating point*> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x \{.*\}> .*",
        "<d x struct> operation",
        "<d x struct> operation",
    ],
    [
        "<%ID> = shufflevector <\d+ x \{.*\}\*> .*",
        "<d x struct*> operation",
        "<d x struct*> operation",
    ],
    [
        "<%ID> = bitcast void \(.* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque) .* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\* .* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\* .* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\* .* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast \[\d+.* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        "<%ID> = bitcast <\d+.* to .*",
        "in-between operation",
        "in-between operation",
    ],
    [
        '<%ID> = bitcast (%"|<%|<?{).* to .*',
        "in-between operation",
        "in-between operation",
    ],
    ["<%ID> = fpext .*", "in-between operation", "in-between operation"],
    ["<%ID> = fptrunc .*", "in-between operation", "in-between operation"],
    ["<%ID> = sext .*", "in-between operation", "in-between operation"],
    ["<%ID> = trunc .* to .*", "in-between operation", "in-between operation"],
    ["<%ID> = zext .*", "in-between operation", "in-between operation"],
    ["<%ID> = sitofp .*", "in-between operation", "in-between operation"],
    ["<%ID> = uitofp .*", "in-between operation", "in-between operation"],
    ["<%ID> = inttoptr .*", "in-between operation", "in-between operation"],
    ["<%ID> = ptrtoint .*", "in-between operation", "in-between operation"],
    ["<%ID> = fptosi .*", "in-between operation", "in-between operation"],
    ["<%ID> = fptoui .*", "in-between operation", "in-between operation"],
    ["<%ID> = extractvalue .*", "in-between operation", "in-between operation"],
    ["<%ID> = insertvalue .*", "in-between operation", "in-between operation"],
    ["resume .*", "in-between operation", "in-between operation"],
    ["(tail |musttail |notail )?call( \w+)? void .*", "call void", "call void"],
    ["i\d{1,2} <(INT|FLOAT)>, label <%ID>", "blob", "blob"],
    ["<%ID> = select .*", "blob", "blob"],
    [".*to label.*unwind label.*", "blob", "blob"],
    ["catch .*", "blob", "blob"],
    ["cleanup", "blob", "blob"],
    ["<%ID> = landingpad .", "blob", "blob"],
    ["; <label>:<LABEL>", "blob", "blob"],
    ["<LABEL>:", "blob", "blob"],
    ["br i1 .*", "blob", "blob"],
    ["br label .*", "blob", "blob"],
    ["indirectbr .*", "blob", "blob"],
    ["switch .*", "blob", "blob"],
    ["unreachable.*", "blob", "blob"],
    ["ret void", "blob", "blob"],
    ["!UNK", "blob", "blob"],
]
