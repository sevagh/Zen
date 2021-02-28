cmake_minimum_required (VERSION 3.14)

# ------------------------------------------------------------------------------#
# MY PERSONAL CONFIG, until i find a better place to put this...
# ------------------------------------------------------------------------------#

set(CMAKE_CXX_COMPILER $ENV{HOME}/GCC-8.4.0/bin/g++)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(IPP_ROOT $ENV{HOME}/intel/ipp CACHE INTERNAL "ipp location hint")
set(VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor CACHE INTERNAL "vendored libaries")

# ------------------------------------------------------------------------------#
# TOP-LEVEL PROJECT
# ------------------------------------------------------------------------------#

project(ZEN LANGUAGES CXX CUDA)
set(VERSION 0.0.1)

include(GNUInstallDirs)

enable_testing()

# ------------------------------------------------------------------------------#
# CONFIGURABLE BUILD OPTIONS
# ------------------------------------------------------------------------------#

option(ENABLE_UBSAN "enable undefined behaviour sanitizer" OFF)
option(ENABLE_ASAN "enable address sanitizer (in place of valgrind)" OFF)
#set(ENV{ASAN_OPTIONS} "protect_shadow_gap=0:replace_intrin=0:detect_leaks=0")

# ------------------------------------------------------------------------------#
# COMMON PREAMBLE
# ------------------------------------------------------------------------------#

set(CUDAToolkit_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} CACHE INTERNAL "needed for findcudatoolkit")
set(CMAKE_CXX_STANDARD 14)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-Wundef")
add_definitions("-Wunreachable-code")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--default-stream-per-thread")
set(CMAKE_CUDA_FLAGS"-std=c++14 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -pthread")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(FindPkgConfig)

# ------------------------------------------------------------------------------#
# SUBPROJECTS
# ------------------------------------------------------------------------------#

add_subdirectory(libzen)
add_subdirectory(zen)
add_subdirectory(demos/pitch-tracking)
add_subdirectory(demos/beat-tracking)

# ------------------------------------------------------------------------------#
# CLANG-FORMAT
# ------------------------------------------------------------------------------#

file(
  GLOB_RECURSE
  ALL_CXX_SOURCE_FILES
  libzen/*.cu
  libzen/*.h
  libzen/libzen/*.h
  zen/*.cu
  zen/*.h
  pitch-tracking/*.cpp
  pitch-tracking/*.cu
  pitch-tracking/*.h
  beat-tracking/*.cpp
  beat-tracking/*.cu
  beat-tracking/*.h)

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(clang-format COMMAND clang-format -i -style=file
                                         ${ALL_CXX_SOURCE_FILES})
endif()
