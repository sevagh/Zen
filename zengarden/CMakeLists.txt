cmake_minimum_required(VERSION 3.14)
project(zengarden LANGUAGES CXX CUDA)
set(VERSION 0.0.1)

set(CUDAToolkit_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CXX_STANDARD 14)
set(VENDOR_DIR ${PROJECT_SOURCE_DIR}/../vendor)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-Wundef")
add_definitions("-Wunreachable-code")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--default-stream-per-thread")
set(CMAKE_CUDA_FLAGS"-std=c++14 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -pthread")

# ------------------------------------------------------------------------------#
# DEPENDENCIES
# ------------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

include(FindPkgConfig)

#find_package(Zengarden REQUIRED)
find_package(Nyquist REQUIRED)
find_package(SoundIo REQUIRED)
find_package(IPP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
	"${PROJECT_SOURCE_DIR}"
	"${PROJECT_SOURCE_DIR}/../libzengarden"
	${SOUNDIO_INCLUDE_DIR}
	${NYQUIST_INCLUDE_DIR}
	${VENDOR_DIR}/clipp/include
)

# ------------------------------------------------------------------------------#
# ZENGARDEN CLI
# ------------------------------------------------------------------------------#

set(ZENGARDEN_SOURCES main.cu realtime.cu)
set(ZENGARDEN_DEPS "${PROJECT_SOURCE_DIR}/../build/build-lib/liblibzengarden.so" ${SOUNDIO_LIBRARY} libnyquist libopus libwavpack CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("ZENGARDEN: Building cli tool")
add_executable(zengarden ${ZENGARDEN_SOURCES})
target_link_libraries(zengarden ${ZENGARDEN_DEPS})
