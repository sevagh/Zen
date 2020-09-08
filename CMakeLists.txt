# ------------------------------------------------------------------------------#
# CMakeLists.txt based on
# https://github.com/sevagh/libmetro/blob/master/CMakeLists.txt
# ------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.14)
project(rhythm_toolkit LANGUAGES CXX CUDA)
set(VERSION 0.0.1)

enable_testing()

include_directory(libzengarden)

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
# INSTALL
# ------------------------------------------------------------------------------#

install(
  TARGETS rhythm_toolkit
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})

install(FILES include/rhythm_toolkit.h DESTINATION include/rhythm_toolkit)

# ------------------------------------------------------------------------------#
# CLANG_FORMAT, CPPCLEAN
# ------------------------------------------------------------------------------#

file(
  GLOB_RECURSE
  ALL_CXX_SOURCE_FILES
  src/*.cpp
  src/*.cu
  src/*.h
  include/*.h
  test/*.cu
  test/*.cpp
  test/*.h
  example/*.cpp
  example/*.cu
  example/*.h
  )

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(clang-format COMMAND clang-format -i -style=file
                                         ${ALL_CXX_SOURCE_FILES})
endif()

# Adding cppclean target if executable is found
# TODO fix this later
find_program(CPP_CLEAN "cppclean")
if(CPP_CLEAN)
  add_custom_target(
    cpp-clean
    COMMAND
      cppclean --include-path ../include/ --include-path ../src/ --include-path
      ${SOUNDIO_INCLUDE_DIR} --include-path ${STK_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
endif()
