#include <complex>
#include <cstddef>
#include <vector>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "npp.h"
#include "nppdefs.h"
#include "nppi.h"

void x_axis_example_right();
void x_axis_example_middle();
void x_axis_example_left();

int main(int argc, char **argv) {
	std::cout << "x-axis median filter extending rightward" << std::endl;
	x_axis_example_right();
	std::cout << std::endl;

	std::cout << "x-axis median filter extending left and right" << std::endl;
	x_axis_example_middle();
	std::cout << std::endl;

	std::cout << "x-axis median filter extending leftward" << std::endl;
	x_axis_example_left();
	std::cout << std::endl;
}

void x_axis_example_right() {
	int width = 9;
	int height = 9;
	int filter_size = 3;

	// use a sliding window starting at the beginning of the window
	// e.g. [(i,j)  (i+1,j)  (i+2,j)] for each median
	NppiSize filter_mask{filter_size, 1};
	NppiPoint anchor{0, 0};

	// as a result, reduce roi by filter_len to not go out of bounds near the end of the matrix
	NppiSize roi{width-filter_size, height};

	// since we're starting from 0 and the filter box points ahead, no need to offset our start position
	// no risk of -1,-1 being accessed
	int start_pixel_offset = 0;

	int nstep = width*sizeof(Npp32f);

	thrust::device_vector<float> in(width*height);
	thrust::device_vector<float> out(width*height);

	Npp32f *in_ptr = (Npp32f*)thrust::raw_pointer_cast(in.data());
	Npp32f *out_ptr = (Npp32f*)thrust::raw_pointer_cast(out.data());

	// fill with 5s across middle row and 8s down middle column
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			if (i == width/2)
				in[i*height + j] = 5;
			if (j == height/2)
				in[i*height + j] = 8;
		}
	}

	Npp8u* buffer;
	Npp32u buffer_size;

	NppStatus npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
	    roi, filter_mask, &buffer_size);
	if (npp_status != NPP_NO_ERROR) {
		std::cerr << "NPP error " << npp_status << std::endl;
		std::exit(1);
	}

	cudaError cuda_status = cudaMalloc(( void** )&buffer, buffer_size);
	if (cuda_status != cudaSuccess) {
		std::cerr << cudaGetErrorString(cuda_status);
		std::exit(1);
	}

	std::cout << "before" << std::endl;
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			std::cout << in[i*height + j] << " ";
		}
		std::cout << std::endl;
	}

	nppiFilterMedian_32f_C1R(in_ptr + start_pixel_offset, nstep,
				 out_ptr + start_pixel_offset, nstep, roi,
				 filter_mask, anchor, buffer);

	std::cout << "after" << std::endl;
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			std::cout << out[i*height + j] << " ";
		}
		std::cout << std::endl;
	}
}

void x_axis_example_middle() {
	int width = 9;
	int height = 9;
	int filter_size = 3;
	int filter_mid = filter_size/2;

	// use a sliding window starting at the middle of the window
	// e.g. [(i-1,j)  (i,j)  (i+1,j)] for each median
	NppiSize filter_mask{filter_size, 1};
	NppiPoint anchor{filter_mid, 0};

	// as a result, reduce roi by filter_len to not go out of bounds near the end of the matrix
	NppiSize roi{width-filter_size, height};

	// since we're starting from 0 and the filter box points backwards and ahead, need to offset our start position
	// by filter_mid
	int start_pixel_offset = filter_mid;

	int nstep = width*sizeof(Npp32f);

	thrust::device_vector<float> in(width*height);
	thrust::device_vector<float> out(width*height);

	Npp32f *in_ptr = (Npp32f*)thrust::raw_pointer_cast(in.data());
	Npp32f *out_ptr = (Npp32f*)thrust::raw_pointer_cast(out.data());

	// fill with 5s across middle row and 8s down middle column
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			if (i == width/2)
				in[i*height + j] = 5;
			if (j == height/2)
				in[i*height + j] = 8;
		}
	}

	Npp8u* buffer;
	Npp32u buffer_size;

	NppStatus npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
	    roi, filter_mask, &buffer_size);
	if (npp_status != NPP_NO_ERROR) {
		std::cerr << "NPP error " << npp_status << std::endl;
		std::exit(1);
	}

	cudaError cuda_status = cudaMalloc(( void** )&buffer, buffer_size);
	if (cuda_status != cudaSuccess) {
		std::cerr << cudaGetErrorString(cuda_status);
		std::exit(1);
	}

	nppiFilterMedian_32f_C1R(in_ptr + start_pixel_offset, nstep,
				 out_ptr + start_pixel_offset, nstep, roi,
				 filter_mask, anchor, buffer);

	std::cout << "after" << std::endl;
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			std::cout << out[i*height + j] << " ";
		}
		std::cout << std::endl;
	}
}

void x_axis_example_left() {
	int width = 9;
	int height = 9;
	int filter_size = 3;

	// use a sliding window starting at the end of the window
	// e.g. [(i-2,j)  (i-1,j)  (i,j)] for each median
	NppiSize filter_mask{filter_size, 1};
	NppiPoint anchor{filter_size, 0};

	// as a result, reduce roi by filter_len to not go out of bounds near the end of the matrix
	NppiSize roi{width-filter_size, height};

	// since we're starting from 0 and the filter box points leftwards, need to offset our start position
	// by the full filter size
	int start_pixel_offset = filter_size;

	int nstep = width*sizeof(Npp32f);

	thrust::device_vector<float> in(width*height);
	thrust::device_vector<float> out(width*height);

	Npp32f *in_ptr = (Npp32f*)thrust::raw_pointer_cast(in.data());
	Npp32f *out_ptr = (Npp32f*)thrust::raw_pointer_cast(out.data());

	// fill with 5s across middle row and 8s down middle column
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			if (i == width/2)
				in[i*height + j] = 5;
			if (j == height/2)
				in[i*height + j] = 8;
		}
	}

	Npp8u* buffer;
	Npp32u buffer_size;

	NppStatus npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
	    roi, filter_mask, &buffer_size);
	if (npp_status != NPP_NO_ERROR) {
		std::cerr << "NPP error " << npp_status << std::endl;
		std::exit(1);
	}

	cudaError cuda_status = cudaMalloc(( void** )&buffer, buffer_size);
	if (cuda_status != cudaSuccess) {
		std::cerr << cudaGetErrorString(cuda_status);
		std::exit(1);
	}

	nppiFilterMedian_32f_C1R(in_ptr + start_pixel_offset, nstep,
				 out_ptr + start_pixel_offset, nstep, roi,
				 filter_mask, anchor, buffer);

	std::cout << "after" << std::endl;
	for (std::size_t i = 0; i < width; ++i) {
		for (std::size_t j = 0; j < height; ++j) {
			std::cout << out[i*height + j] << " ";
		}
		std::cout << std::endl;
	}
}

