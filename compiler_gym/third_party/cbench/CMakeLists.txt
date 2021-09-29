# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_genrule(
  NAME cbench_tar
  DEPENDS ::cbench
  OUTS "llvm_bitcodes-10.0.0-cbench-v1.tar.bz2"
  CMD "tar cjfh \"$@\" -C \"$(@D)\" cbench-v1"
  PUBLIC
)

cg_filegroup(
  NAME cbench
  DEPENDS
    ::adpcm
    ::bitcount
    ::blowfish
    ::bzip2
    ::crc32
    ::dijkstra
    ::ghostscript
    ::gsm
    ::ispell
    ::jpeg-c
    ::jpeg-d
    ::lame
    ::patricia
    ::qsort
    ::rijndael
    ::sha
    ::stringsearch
    ::stringsearch2
    ::susan
    ::tiff2bw
    ::tiff2rgba
    ::tiffdither
    ::tiffmedian
  PUBLIC
)

set(cBench_RUNTIME_DATA_SRC_FILES
  ${cBench_consumer_tiff_data_FILE}
  ${cBench_office_data_FILE}
  ${cBench_telecom_data_FILE}
  ${cBench_consumer_jpeg_data_FILE}
  ${cBench_telecom_gsm_data_FILE}
  ${cBench_consumer_data_FILE}
  ${cBench_bzip2_data_FILE}
  ${cBench_network_patricia_data_FILE}
  ${cBench_network_dijkstra_data_FILE}
  ${cBench_automotive_susan_data_FILE}
  ${cBench_automotive_qsort_data_FILE})
set(_CMD "cp -R \"${CMAKE_CURRENT_SOURCE_DIR}/runtime_data\" \"$@\"")
foreach(FILE_ IN LISTS cBench_RUNTIME_DATA_SRC_FILES)
  string(CONCAT _CMD "${_CMD}" " && tar xzf \"${FILE_}\" -C \"$(@D)/runtime_data\"")
endforeach()
file(GLOB_RECURSE LOCAL_RUNTIME_DATA_FILES
  LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/runtime_data")
cg_genrule(
  NAME make_runtime_data
  SRCS
    ${cBench_RUNTIME_DATA_SRC_FILES}
    ${LOCAL_RUNTIME_DATA_FILES}
  OUTS "runtime_data"
  COMMAND "${_CMD}"
)

cg_filegroup(
  NAME benchmarks_list
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks.txt"
  PUBLIC
)

cg_py_binary(
  NAME make_llvm_module
  SRCS "make_llvm_module.py"
  DEPS compiler_gym::envs::llvm::llvm_benchmark
)

set(CBENCH_MODULES
  crc32
  jpeg-c
  jpeg-d
  stringsearch2
  adpcm
  bitcount
  blowfish
  bzip2
  dijkstra
  patricia
  qsort
  rijndael
  sha
  stringsearch
  susan
  tiff2bw
  tiff2rgba
  tiffdither
  tiffmedian)
set(CBENCH_MODULE_DIRS
  "${cBench_SRC_DIR}/telecom_CRC32"
  "${ctuning_ai_SRC_DIR}/program/cbench-consumer-jpeg-c"
  "${ctuning_ai_SRC_DIR}/program/cbench-consumer-jpeg-d"
  "${ctuning_ai_SRC_DIR}/program/cbench-office-stringsearch2"
  "${cBench_SRC_DIR}/telecom_adpcm_c"
  "${cBench_SRC_DIR}/automotive_bitcount"
  "${cBench_SRC_DIR}/security_blowfish_d"
  "${cBench_SRC_DIR}/bzip2d"
  "${cBench_SRC_DIR}/network_dijkstra"
  "${cBench_SRC_DIR}/network_patricia"
  "${cBench_SRC_DIR}/automotive_qsort1"
  "${cBench_SRC_DIR}/security_rijndael_d"
  "${cBench_SRC_DIR}/security_sha"
  "${cBench_SRC_DIR}/office_stringsearch1"
  "${cBench_SRC_DIR}/automotive_susan_c"
  "${cBench_SRC_DIR}/consumer_tiff2bw"
  "${cBench_SRC_DIR}/consumer_tiff2rgba"
  "${cBench_SRC_DIR}/consumer_tiffdither"
  "${cBench_SRC_DIR}/consumer_tiffmedian")
foreach(MODULE_ DIR_ IN ZIP_LISTS CBENCH_MODULES CBENCH_MODULE_DIRS)
  file(GLOB_RECURSE MODULE_FILES LIST_DIRECTORIES true "${DIR_}/*")
  string(CONCAT _CMD_PY
    "\"${Python3_EXECUTABLE}\" "
      "\"${CMAKE_CURRENT_BINARY_DIR}/make_llvm_module.py\" \"${DIR_}\" \"$@\"")
  set_command_pythonpath(COMMAND "${_CMD_PY}" RESULT _CMD_PY)
  string(CONCAT _CMD
    "mkdir -p \"$(@D)\" && "
    "${_CMD_PY}")
  cg_genrule(
    NAME ${MODULE_}
    SRCS
      ${MODULE_FILES}
    OUTS "cbench-v1/${MODULE_}.bc"
    COMMAND "${_CMD}"
    DEPENDS
      ::make_llvm_module
    PUBLIC
  )
endforeach()

file(GLOB_RECURSE office_ghostscript_FILES LIST_DIRECTORIES true
  "${cBench_SRC_DIR}/office_ghostscript/*")
string(CONCAT _CMD
  "mkdir -p \"$(@D)\" && "
  "rsync -rL \"${cBench_SRC_DIR}/office_ghostscript/\" \"$(@D)/office_ghostscript_src/\" && "
  "patch --quiet --forward \"$(@D)/office_ghostscript_src/src/idebug.c\" < \"${CMAKE_CURRENT_SOURCE_DIR}/cBench-ghostscript-idebug.c.patch\" && "
  "patch --quiet --forward \"$(@D)/office_ghostscript_src/src/std.h\" < \"${CMAKE_CURRENT_SOURCE_DIR}/cBench-ghostscript-std.h.patch\" && ")
string(CONCAT _CMD_PY
  "\"${Python3_EXECUTABLE}\" "
    "\"${CMAKE_CURRENT_BINARY_DIR}/make_llvm_module.py\" \"$(@D)/office_ghostscript_src\" \"$@\"")
set_command_pythonpath(COMMAND "${_CMD_PY}" RESULT _CMD_PY)
string(CONCAT _CMD "${_CMD}" "${_CMD_PY}")
cg_genrule(
  NAME ghostscript
  SRCS
    ${office_ghostscript_FILES}
    "cBench-ghostscript-std.h.patch"
    "cBench-ghostscript-idebug.c.patch"
  OUTS "cbench-v1/ghostscript.bc"
  COMMAND "${_CMD}"
  DEPENDS
    ::make_llvm_module
  PUBLIC
)

file(GLOB_RECURSE telecom_gsm_FILES LIST_DIRECTORIES true
  "${cBench_SRC_DIR}/telecom_gsm/*")
string(CONCAT _CMD
  "mkdir -p \"$(@D)\" && "
  "rsync -rL \"${cBench_SRC_DIR}/telecom_gsm/\" \"$(@D)/telecom_gsm_src/\" && "
  "patch --quiet --forward \"$(@D)/telecom_gsm_src/src/add.c\""
    " < \"${CMAKE_CURRENT_SOURCE_DIR}/cBench-gsm-add.c.patch\" && ")
string(CONCAT _CMD_PY
  "\"${Python3_EXECUTABLE}\" "
    "\"${CMAKE_CURRENT_BINARY_DIR}/make_llvm_module.py\" \"$(@D)/telecom_gsm_src\" \"$@\" "
    "-DSASR -DSTUPID_COMPILER -DNeedFunctionPrototypes=1")
set_command_pythonpath(COMMAND "${_CMD_PY}" RESULT _CMD_PY)
string(CONCAT _CMD "${_CMD}" "${_CMD_PY}")
cg_genrule(
  NAME gsm
  SRCS
    ${telecom_gsm_FILES}
    "cBench-gsm-add.c.patch"
  OUTS "cbench-v1/gsm.bc"
  COMMAND "${_CMD}"
  DEPENDS
    ::make_llvm_module
  PUBLIC
)

file(GLOB_RECURSE office_ispell_FILES LIST_DIRECTORIES true
  "${cBench_SRC_DIR}/office_ispell/*")
string(CONCAT _CMD
  "mkdir -p \"$(@D)\" && "
  "rsync -rL \"${cBench_SRC_DIR}/office_ispell/\" \"$(@D)/office_ispell_src/\" && "
  "patch --quiet --forward \"$(@D)/office_ispell_src/src/correct.c\" < \"${CMAKE_CURRENT_SOURCE_DIR}/cBench-ispell-correct.c.patch\" && ")
string(CONCAT _CMD_PY
  "\"${Python3_EXECUTABLE}\" "
    "\"${CMAKE_CURRENT_BINARY_DIR}/make_llvm_module.py\" \"$(@D)/office_ispell_src\" \"$@\"")
set_command_pythonpath(COMMAND "${_CMD_PY}" RESULT _CMD_PY)
string(CONCAT _CMD "${_CMD}" "${_CMD_PY}")
cg_genrule(
  NAME ispell
  SRCS
    ${office_ispell_FILES}
    "cBench-ispell-correct.c.patch"
  OUTS "cbench-v1/ispell.bc"
  COMMAND "${_CMD}"
  DEPENDS
    ::make_llvm_module
  PUBLIC
)

file(GLOB_RECURSE consumer_lame_FILES LIST_DIRECTORIES true
  "${cBench_SRC_DIR}/consumer_lame/*")
string(CONCAT _CMD
  "mkdir -p \"$(@D)\" && ")
string(CONCAT _CMD_PY
  "\"${Python3_EXECUTABLE}\" "
    "\"${CMAKE_CURRENT_BINARY_DIR}/make_llvm_module.py\" "
    "\"${cBench_SRC_DIR}/consumer_lame\" \"$@\" -DLAMESNDFILE -DHAVEMPGLIB -DLAMEPARSE")
set_command_pythonpath(COMMAND "${_CMD_PY}" RESULT _CMD_PY)
string(CONCAT _CMD "${_CMD}" "${_CMD_PY}")
cg_genrule(
  NAME lame
  SRCS
    ${consumer_lame_FILES}
  OUTS "cbench-v1/lame.bc"
  COMMAND "${_CMD}"
  DEPENDS
    ::make_llvm_module
  PUBLIC
)
