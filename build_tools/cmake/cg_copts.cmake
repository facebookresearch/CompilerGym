# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# C/C++ options as used within Compiler Gym
#-------------------------------------------------------------------------------
#
#         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
#         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
#         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
#         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
#          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
#
# Everything here is added to *every* cg_cc_library/cg_cc_binary/etc.
# That includes both runtime and compiler components, and these may propagate
# out to user code interacting with either (such as custom modules).
#
# Be extremely judicious in the use of these flags.
#
# - Need to disable a warning?
#   Usually these are encountered in compiler-specific code and can be disabled
#   in a compiler-specific way. Only add global warning disables when it's clear
#   that we never want them or that they'll show up in a lot of places.
#
#   See: https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code
#
# - Need to add a linker dependency?
#   First figure out if you *really* need it. If it's only required on specific
#   platforms and in very specific files clang or msvc are used prefer
#   autolinking. GCC is stubborn and doesn't have autolinking so additional
#   flags may be required there.
#
#   See: https://en.wikipedia.org/wiki/Auto-linking


set(COMPILER_GYM_CXX_STANDARD ${CMAKE_CXX_STANDARD})

# TODO(benvanik): fix these names (or remove entirely).
set(COMPILER_GYM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_GYM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_GYM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Compiler diagnostics.
cg_select_compiler_opts(COMPILER_GYM_DEFAULT_COPTS
  # Clang diagnostics. These largely match the set of warnings used within
  # Google. They have not been audited super carefully by the IREE team but are
  # generally thought to be a good set and consistency with those used
  # internally is very useful when importing. If you feel that some of these
  # should be different (especially more strict), please raise an issue!
  CLANG
    "-Werror"
    "-Wall"

    # Disable warnings we don't care about or that generally have a low
    # signal/noise ratio.
    "-Wno-ambiguous-member-template"
    "-Wno-char-subscripts"
    "-Wno-deprecated-declarations"
    "-Wno-extern-c-compat" # Matches upstream. Cannot impact due to extern C inclusion method.
    "-Wno-gnu-alignof-expression"
    "-Wno-gnu-variable-sized-type-not-at-end"
    "-Wno-ignored-optimization-argument"
    "-Wno-invalid-offsetof" # Technically UB but needed for intrusive ptrs
    "-Wno-invalid-source-encoding"
    "-Wno-mismatched-tags"
    "-Wno-pointer-sign"
    "-Wno-reserved-user-defined-literal"
    "-Wno-return-type-c-linkage"
    "-Wno-self-assign-overloaded"
    "-Wno-sign-compare"
    "-Wno-signed-unsigned-wchar"
    "-Wno-strict-overflow"
    "-Wno-trigraphs"
    "-Wno-unknown-pragmas"
    "-Wno-unknown-warning-option"
    "-Wno-unused-command-line-argument"
    "-Wno-unused-const-variable"
    "-Wno-unused-function"
    "-Wno-unused-local-typedef"
    "-Wno-unused-private-field"
    "-Wno-user-defined-warnings"

    # Explicitly enable some additional warnings.
    # Some of these aren't on by default, or under -Wall, or are subsets of
    # warnings turned off above.
    "-Wctad-maybe-unsupported"
    "-Wfloat-overflow-conversion"
    "-Wfloat-zero-conversion"
    "-Wfor-loop-analysis"
    "-Wformat-security"
    "-Wgnu-redeclared-enum"
    "-Wimplicit-fallthrough"
    "-Winfinite-recursion"
    "-Wliteral-conversion"
    #"-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wself-assign"
    "-Wstring-conversion"
    "-Wtautological-overlap-compare"
    "-Wthread-safety"
    "-Wthread-safety-beta"
    "-Wunused-comparison"
    "-Wvla"

  # TODO(#6959): Enable -Werror once we have a presubmit CI.
  GCC
    "-Wall"
    "-Wno-address-of-packed-member"
    "-Wno-comment"
    "-Wno-format-zero-length"
    # Technically UB but needed for intrusive ptrs
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-invalid-offsetof>
    $<$<COMPILE_LANGUAGE:C>:-Wno-pointer-sign>
    "-Wno-sign-compare"
    "-Wno-unused-function"
)
