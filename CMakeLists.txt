# ------------------------------------------------------------------------------#
# CMakeLists.txt based on
# https://github.com/sevagh/libmetro/blob/master/CMakeLists.txt
# ------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.14)
project(rhythm_toolkit LANGUAGES CXX CUDA)
set(VERSION 0.0.1)

enable_testing()

# ------------------------------------------------------------------------------#
# BASICS
# ------------------------------------------------------------------------------#

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(VENDOR_DIR ${PROJECT_SOURCE_DIR}/vendor)

set(CMAKE_CXX_STANDARD 20)

set(BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-Wundef")
add_definitions("-Wunreachable-code")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -pthread")

option(BUILD_STATIC "Build static library" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# ------------------------------------------------------------------------------#
# DEPS
# ------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(FindPkgConfig)

find_package(GTest REQUIRED)
find_package(gflags REQUIRED)
find_package(FFTS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Nyquist REQUIRED)

# ------------------------------------------------------------------------------#
# INCLUDE DIRS
# ------------------------------------------------------------------------------#

include_directories(
  "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include"
  ${GTEST_INCLUDE_DIRS} ${FFTS_INCLUDE_DIRS} ${VENDOR_DIR}/mf2d/src)

# ------------------------------------------------------------------------------#
# VENDORED LIBS
# ------------------------------------------------------------------------------#

set(MF2D_SOURCES ${VENDOR_DIR}/mf2d/src/filter.cc)

add_library(mf2d SHARED ${MF2D_SOURCES})
target_link_libraries(mf2d PUBLIC OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------------------#
# LIBS
# ------------------------------------------------------------------------------#

include_directories(
  "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include"
  ${GTEST_INCLUDE_DIRS} ${FFTS_INCLUDE_DIRS} ${VENDOR_DIR}/mf2d/src ${VENDOR_DIR}/libnyquist/include)
link_directories(${GTEST_LIBRARY_DIRS})

set(RHYTHM_TOOLKIT_SOURCES src/rhythm_toolkit.cpp src/hpss.cpp)

set(RHYTHM_TOOLKIT_DEPS rhythm_toolkit ${FFTS_LIBRARY})

if(BUILD_STATIC)
  message("LIBRHYTHM_TOOLKIT: Building static library")
  add_library(rhythm_toolkit STATIC ${RHYTHM_TOOLKIT_SOURCES})
else()
  message("LIBRHYTHM_TOOLKIT: Building dynamic library")
  add_library(rhythm_toolkit SHARED ${RHYTHM_TOOLKIT_SOURCES})
endif()

target_link_libraries(rhythm_toolkit ${SOUNDIO_LIBRARY} ${STK_LIBRARY} mf2d)

# ------------------------------------------------------------------------------#
# CLANG-TIDY
# ------------------------------------------------------------------------------#

find_program(
  CLANG_TIDY_EXE
  NAMES "clang-tidy"
  DOC "Path to clang-tidy executable")
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-clang-analyzer-alpha.*")
endif()

option(ENABLE_CLANG_TIDY "compile with clang-tidy" OFF)
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
  message("CLANG_TIDY: ENABLED")
  set_target_properties(rhythm_toolkit PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
else()
  message("CLANG_TIDY: DISABLED")
endif()

# ------------------------------------------------------------------------------#
# BINS
# ------------------------------------------------------------------------------#

function(rhythm_toolkit_example name ext)
  add_executable(example_${name} example/${name}.${ext})
  target_link_libraries(example_${name} ${RHYTHM_TOOLKIT_DEPS} gflags libnyquist libwavpack libopus)
endfunction()

if(BUILD_EXAMPLES)
  message("LIBRHYTHM_TOOLKIT: Building examples")
  rhythm_toolkit_example(wav_hpss cpp)
  rhythm_toolkit_example(cuda_test cu)
  #rhythm_toolkit_example(dummy_hpss)
  #rhythm_toolkit_example(ffts_test)
else()
  message("LIBRHYTHM_TOOLKIT: Skipping examples")
endif()

# ------------------------------------------------------------------------------#
# INSTALL
# ------------------------------------------------------------------------------#

install(
  TARGETS rhythm_toolkit
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})

install(FILES include/rhythm_toolkit.h DESTINATION include/rhythm_toolkit)

# ------------------------------------------------------------------------------#
# TESTS
# ------------------------------------------------------------------------------#

set(RHYTHM_TOOLKIT_TESTS_DEPS ${RHYTHM_TOOLKIT_DEPS} gtest gtest_main pthread)

find_program(VALGRIND_EXEC "valgrind")
if(VALGRIND_EXEC)
  message("VALGRIND: ${VALGRIND_EXEC}")
  set(VALGRIND_ARGS --leak-check=full --track-origins=yes)
else()
  message("VALGRIND: DISABLED - install valgrind to enable")
endif()

function(rhythm_toolkit_gtest name)
  add_executable(test_${name} test/${name}.cpp)
  target_link_libraries(test_${name} ${RHYTHM_TOOLKIT_TESTS_DEPS})
  target_compile_definitions(test_${name} PUBLIC UNIT_TESTS)

  add_test(test_${name} ${BIN_DIR}/test_${name})
  set_tests_properties(test_${name} PROPERTIES LABELS "test")

  if(VALGRIND_EXEC)
    add_test(${name}_valgrind ${VALGRIND_EXEC} ${VALGRIND_ARGS}
             ${BIN_DIR}/test_${name})
    set_tests_properties(${name}_valgrind PROPERTIES LABELS "valgrind")
  endif()
endfunction()

if(BUILD_TESTS)
  message("LIBRHYTHM_TOOLKIT: Building tests")
  rhythm_toolkit_gtest(rhythm_toolkit)
else()
  message("LIBRHYTHM_TOOLKIT: Skipping tests")
endif()

# ------------------------------------------------------------------------------#
# UBSAN
# ------------------------------------------------------------------------------#

option(ENABLE_UBSAN
       "enable undefined behaviour sanitizer (affects performance)" OFF)
if(ENABLE_UBSAN)
  message("UBSAN: ENABLED")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=address -lubsan")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined -fsanitize=address -lubsan"
  )
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=undefined -fsanitize=address -lubsan"
  )
else()
  message("UBSAN: DISABLED")
endif()

# ------------------------------------------------------------------------------#
# CLANG_FORMAT, CPPCLEAN
# ------------------------------------------------------------------------------#

file(
  GLOB_RECURSE
  ALL_CXX_SOURCE_FILES
  src/*.cpp
  src/*.h
  include/*.h
  test/*.cpp
  examples/*.cpp)

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(clang-format COMMAND clang-format -i -style=file
                                         ${ALL_CXX_SOURCE_FILES})
endif()

# Adding cppclean target if executable is found
find_program(CPP_CLEAN "cppclean")
if(CPP_CLEAN)
  add_custom_target(
    cpp-clean
    COMMAND
      cppclean --include-path ../include/ --include-path ../src/ --include-path
      ${SOUNDIO_INCLUDE_DIR} --include-path ${STK_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ------------------------------------------------------------------------------#
# DOXYGEN DOCS
# ------------------------------------------------------------------------------#

# check if Doxygen is installed
find_package(Doxygen)
if(DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("Doxygen build started")

  # note the option ALL which allows to build the docs together with the
  # application
  add_custom_target(
    doc-doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()
