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
	${NYQUIST_INCLUDE_DIR}
	${VENDOR_DIR}/clipp/include
)

# ------------------------------------------------------------------------------#
# ZEN CLI
# ------------------------------------------------------------------------------#

set(ZEN_SOURCES main.cu)
set(ZEN_DEPS libzen libnyquist libopus libwavpack CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("ZEN: Building cli tool")
add_executable(zen ${ZEN_SOURCES})
target_link_libraries(zen LINK_PUBLIC ${ZEN_DEPS})
