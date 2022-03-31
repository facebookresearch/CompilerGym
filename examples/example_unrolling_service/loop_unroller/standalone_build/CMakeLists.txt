cmake_minimum_required(VERSION 3.14.0)
project(loop_unroller)

#===============================================================================
# 1. VERIFY LLVM INSTALLATION DIR
#===============================================================================
# Set this to a valid LLVM installation dir
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# A bit of a sanity checking
set(LT_LLVM_INCLUDE_DIR "${LT_LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LT_LLVM_INCLUDE_DIR}")
    message(
        FATAL_ERROR
        " LT_LLVM_INSTALL_DIR (${LT_LLVM_INCLUDE_DIR}) is invalid."
    )
endif()

set(LT_LLVM_CMAKE_FILE "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
if(NOT EXISTS "${LT_LLVM_CMAKE_FILE}")
    message(
        FATAL_ERROR
        " LT_LLVM_CMAKE_FILE (${LT_LLVM_CMAKE_FILE}) is invalid."
    )
endif()

#===============================================================================
# 2. LOAD LLVM CONFIGURATION
#    For more: http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
#===============================================================================
# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

# TODO: find a way to set LLVM 10.0.0 as minimum version
find_package(LLVM 10.0.0 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LT_LLVM_INSTALL_DIR}")

message(
    "LLVM STATUS:
  Definitions ${LLVM_DEFINITIONS}
  Includes    ${LLVM_INCLUDE_DIRS}
  Libraries   ${LLVM_LIBRARY_DIRS}
  Targets     ${LLVM_TARGETS_TO_BUILD}"
)

# Set the LLVM header and library paths
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#===============================================================================
# 3. LOOP_UNROLLER BUILD CONFIGURATION
#===============================================================================
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (default Debug):" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fdiagnostics-color=always")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# -fvisibility-inlines-hidden is set when building LLVM and on Darwin warnings
# are triggered if llvm-tutor is built without this flag (though otherwise it
# builds fine). For consistency, add it here too.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(
    "-fvisibility-inlines-hidden"
    SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG
)
if(${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG} EQUAL "1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# Set the build directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

##===============================================================================
## 4. ADD SUB-TARGETS
## Doing this at the end so that all definitions and link/include paths are
## available for the sub-projects.
##===============================================================================
add_executable(loop_unroller ../loop_unroller.cc)
target_link_libraries(loop_unroller LLVMPasses)

# On Darwin (unlike on Linux), undefined symbols in shared objects are not
# allowed at the end of the link-edit. The plugins defined here:
#  - _are_ shared objects
#  - reference symbols from LLVM shared libraries, i.e. symbols which are
#    undefined until those shared objects are loaded in memory (and hence
#    _undefined_ during static linking)
# The build will fail with errors like this:
#    "Undefined symbols for architecture x86_64"
# with various LLVM symbols being undefined. Since those symbols are later
# loaded and resolved at runtime, these errors are false positives.
# This behaviour can be modified via the '-undefined' OS X linker flag as
# follows.
target_link_libraries(
    loop_unroller
    "$<$<PLATFORM_ID:Darwin>:-flat_namespace -undefined suppress>" # dynamic_lookup
)
