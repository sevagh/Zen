cmake_minimum_required(VERSION 3.14)
project(libzengarden LANGUAGES CXX CUDA)
set(VERSION 0.0.1)

include(GNUInstallDirs)

enable_testing()

set(CUDAToolkit_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CXX_STANDARD 14)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-Wundef")
add_definitions("-Wunreachable-code")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--default-stream-per-thread")
set(CMAKE_CUDA_FLAGS"-std=c++14 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -pthread")

option(BUILD_DOCS "Build docs" OFF)
option(BUILD_BENCHES "Build benchmarks" OFF)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_CLANG_TIDY "compile with clang-tidy" OFF)

# ------------------------------------------------------------------------------#
# DEPENDENCIES
# ------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

include(FindPkgConfig)

find_package(IPP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
	"${PROJECT_SOURCE_DIR}"
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${IPP_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------#
# LIBZENGARDEN
# ------------------------------------------------------------------------------#

set(LIBZENGARDEN_SOURCES
	${PROJECT_SOURCE_DIR}/hps/hps.cu
	#${PROJECT_SOURCE_DIR}/onset/onset.cu
	)

file(
	GLOB_RECURSE
	LIBZENGARDEN_PUBLIC_HEADERS
	${PROJECT_SOURCE_DIR}/libzengarden/*.h)

set(LIBZENGARDEN_DEPS CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("LIBZENGARDEN: Building dynamic library")
add_library(libzengarden SHARED ${LIBZENGARDEN_SOURCES})
set_target_properties(libzengarden PROPERTIES PUBLIC_HEADER "${LIBZENGARDEN_PUBLIC_HEADERS}")
target_link_libraries(libzengarden ${LIBZENGARDEN_DEPS})

install(TARGETS libzengarden
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libzengarden)

# ------------------------------------------------------------------------------#
# TESTS AND BENCHES
# ------------------------------------------------------------------------------#

find_program(CUMEMCHK_EXEC "cuda-memcheck")
if(CUMEMCHK_EXEC)
  message("CUMEMCHK: ${CUMEMCHK_EXEC}")
else()
  message("CUMEMCHK: DISABLED - install cuda-memcheck to enable")
endif()

function(zg_unittest path name)
  add_executable(${name}.test ${path}/${name}.test.cu)
  target_link_libraries(${name}.test ${LIBZENGARDEN_DEPS} gtest gtest_main pthread -static-libstdc++)

  add_test(${name}_test ${name}.test)
  set_tests_properties(${name}_test PROPERTIES LABELS "test")

  if(CUMEMCHK_EXEC)
    add_test(${name}_cumemchk ${CUMEMCHK_EXEC} ${name}.test)
    set_tests_properties(${name}_cumemchk PROPERTIES LABELS "cuda-memcheck")
  endif()
endfunction()

if(BUILD_TESTS)
	find_package(GTest REQUIRED)
	message("LIBZENGARDEN: Building unit tests")
	
	zg_unittest(hps mfilt)
else()
	message("LIBZENGARDEN: Skipping unit tests")
endif()

function(zg_bench path name)
	add_executable(${name}.bench ${path}/${name}.bench.cu)
	target_link_libraries(${name}.bench ${LIBZENGARDEN_DEPS} benchmark benchmark_main pthread -static-libstdc++)
endfunction()

if(BUILD_BENCHES)
	find_package(benchmark REQUIRED)
	message("LIBZENGARDEN: Building benches")

	zg_bench(hps mfilt)
else()
	message("LIBZENGARDEN: Skipping benches")
endif()

# ------------------------------------------------------------------------------#
# UBSAN
# ------------------------------------------------------------------------------#

option(ENABLE_UBSAN
       "enable undefined behaviour sanitizer" OFF)
if(ENABLE_UBSAN)
	message("UBSAN: ENABLED")
	set(UBSAN_OPTS "-fsanitize=undefined -static-libubsan")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_OPTS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UBSAN_OPTS}")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UBSAN_OPTS}")
else()
	message("UBSAN: DISABLED")
endif()

# ------------------------------------------------------------------------------#
# ASAN
# ------------------------------------------------------------------------------#

option(ENABLE_ASAN
       "enable address sanitizer (in place of valgrind)" OFF)
if(ENABLE_ASAN)
	message("ASAN: ENABLED")
	set(ASAN_OPTS " -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer -static-libasan")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_OPTS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_OPTS}")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_OPTS}")
else()
	message("ASAN: DISABLED")
endif()

# ------------------------------------------------------------------------------#
# DOXYGEN DOCS
# ------------------------------------------------------------------------------#

# check if Doxygen is installed
find_package(Doxygen)
if(BUILD_DOCS AND DOXYGEN_FOUND)
	message("Doxygen found and enabled")
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

# ------------------------------------------------------------------------------#
# CPPCLEAN
# ------------------------------------------------------------------------------#

find_program(CPP_CLEAN "cppclean")
if(CPP_CLEAN)
	message("cppclean found and enabled")
	add_custom_target(
	cpp-clean
	COMMAND
		cppclean
		--include-path "${PROJECT_SOURCE_DIR}"
		--include-path "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
		--include-path "${IPP_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}")
endif()
