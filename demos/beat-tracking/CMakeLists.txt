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
	${VENDOR_DIR}/gcem/include
	${IPP_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------#
# BEAT TRACKING DEMO
# ------------------------------------------------------------------------------#

set(BEAT_TRACKING_SOURCES OnsetDetection.cpp BTrack.cpp main.cu)
set(BEAT_TRACKING_DEPS libzen libnyquist libopus libwavpack CUDA::cufft CUDA::nppif CUDA::nppisu CUDA::npps CUDA::nppidei ${IPP_LIBRARY})

message("ZEN: Building beat tracking tool")
add_executable(beat-track ${BEAT_TRACKING_SOURCES})
target_link_libraries(beat-track LINK_PUBLIC ${BEAT_TRACKING_DEPS})
