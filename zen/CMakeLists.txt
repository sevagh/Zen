message(STATUS "DEBUG IPP_ROOT=${IPP_ROOT}")
message(STATUS "DEBUG CUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}")

# ------------------------------------------------------------------------------#
# DEPENDENCIES
# ------------------------------------------------------------------------------#

set(CUDAToolkit_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_package(Nyquist REQUIRED)
find_package(IPP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/../libzen"
	${SOUNDIO_INCLUDE_DIR}
	${NYQUIST_INCLUDE_DIR}
	${VENDOR_DIR}/clipp/include
)

# ------------------------------------------------------------------------------#
# ZENGARDEN CLI
# ------------------------------------------------------------------------------#

set(ZENGARDEN_SOURCES main.cu)
set(ZENGARDEN_DEPS libzen libnyquist libopus libwavpack CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("ZENGARDEN: Building cli tool")
add_executable(zen ${ZENGARDEN_SOURCES})
target_link_libraries(zen LINK_PUBLIC ${ZENGARDEN_DEPS})
