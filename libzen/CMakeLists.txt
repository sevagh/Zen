message(STATUS "DEBUG IPP_ROOT=${IPP_ROOT}")
message(STATUS "DEBUG CUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}")

# ------------------------------------------------------------------------------#
# CONFIGURABLE BUILD OPTIONS
# ------------------------------------------------------------------------------#

option(BUILD_DOCS "Build docs" OFF)
option(BUILD_BENCHES "Build benchmarks" OFF)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_CLANG_TIDY "compile with clang-tidy" OFF)

# ------------------------------------------------------------------------------#
# DEPENDENCIES
# ------------------------------------------------------------------------------#

find_package(IPP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${IPP_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------#
# LIBZEN
# ------------------------------------------------------------------------------#

set(LIBZEN_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/hps/hps.cu
	${CMAKE_CURRENT_SOURCE_DIR}/core.cu
	)

file(
	GLOB_RECURSE
	LIBZEN_PUBLIC_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/libzen/*.h)

set(LIBZEN_DEPS CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("LIBZEN: Building dynamic library")
add_library(libzen SHARED ${LIBZEN_SOURCES})
set_target_properties(libzen PROPERTIES PUBLIC_HEADER "${LIBZEN_PUBLIC_HEADERS}")
target_link_libraries(libzen ${LIBZEN_DEPS})
target_include_directories(libzen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS libzen
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libzen)

# ------------------------------------------------------------------------------#
# TESTS AND BENCHES
# ------------------------------------------------------------------------------#

find_program(CUMEMCHK_EXEC "cuda-memcheck")
if(CUMEMCHK_EXEC)
  message("CUMEMCHK: ${CUMEMCHK_EXEC}")
  set(CUMEMCHK_ARGS --leak-check full)
else()
  message("CUMEMCHK: DISABLED - install cuda-memcheck to enable")
endif()

function(zen_unittest path name do_cumemchk)
  add_executable(${name}.test ${path}/${name}.test.cu)
  target_link_libraries(${name}.test libzen ${LIBZEN_DEPS} gtest gtest_main pthread -static-libstdc++)

  add_test(${name}_test ${name}.test)
  set_tests_properties(${name}_test PROPERTIES LABELS "test")

  if(CUMEMCHK_EXEC AND do_cumemchk)
	  add_test(${name}_cumemchk ${CUMEMCHK_EXEC} ${CUMEMCHK_ARGS} ${name}.test)
    set_tests_properties(${name}_cumemchk PROPERTIES LABELS "cuda-memcheck")
  endif()
endfunction()

if(BUILD_TESTS)
	find_package(GTest REQUIRED)
	message("LIBZEN: Building unit tests")

	zen_unittest(hps mfilt 1)
	zen_unittest(hps box 1)
	zen_unittest(hps hps 0) # very slow, disable cuda-memcheck
	zen_unittest(. fftw 0) # very slow, disable cuda-memcheck
	zen_unittest(hps hps_gpu_public 0) # e2e test for cuda memcheck
	zen_unittest(hps hps_cpu_public 0) # cpu only, no need for cuda memcheck
else()
	message("LIBZEN: Skipping unit tests")
endif()

function(zen_bench path name)
	add_executable(${name}.bench ${path}/${name}.bench.cu)
	target_link_libraries(${name}.bench libzen ${LIBZEN_DEPS} benchmark benchmark_main pthread -static-libstdc++)
endfunction()

if(BUILD_BENCHES)
	find_package(benchmark REQUIRED)
	message("LIBZEN: Building benches")

	zen_bench(hps mfilt)
	zen_bench(. fftw)
else()
	message("LIBZEN: Skipping benches")
endif()

# ------------------------------------------------------------------------------#
# UBSAN
# ------------------------------------------------------------------------------#

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
		--include-path "${CMAKE_CURRENT_SOURCE_DIR}"
		--include-path "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
		--include-path "${IPP_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
