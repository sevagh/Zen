#ifndef RHYTHM_TOOLKIT_IO_H
#define RHYTHM_TOOLKIT_IO_H

#include <algorithm>
#include <cstddef>
#include <numeric>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "npp.h"

namespace rhythm_toolkit {
namespace io {
	class IO {
	public:
		float* host_in;
		float* host_out;
		thrust::device_ptr<float> device_in;
		thrust::device_ptr<float> device_out;

		IO(std::size_t hop)
		{
			cudaSetDeviceFlags(cudaDeviceMapHost);

			cudaError_t cuda_error;
			unsigned int cuda_malloc_flags = cudaHostAllocMapped;

			// use mapped + wc for performance:
			// http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/online/group__CUDART__MEMORY_g217d441a73d9304c6f0ccc22ec307dba.html
			cuda_error = cudaHostAlloc(
			    ( void** )&host_in, hop * sizeof(float),
			    cuda_malloc_flags | cudaHostAllocWriteCombined);

			if (cuda_error != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostAlloc(
			    ( void** )&host_out, hop * sizeof(float), cuda_malloc_flags);

			if (cuda_error != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostGetDevicePointer(
			    ( void** )(&device_in_raw_ptr), ( void* )host_in, 0);
			if (cuda_error != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostGetDevicePointer(
			    ( void** )(&device_out_raw_ptr), ( void* )host_out, 0);
			if (cuda_error != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			device_in = thrust::device_pointer_cast(device_in_raw_ptr);
			device_out = thrust::device_pointer_cast(device_out_raw_ptr);
		}

		~IO()
		{
			cudaFree(host_in);
			cudaFree(host_out);
		}

	private:
		float* device_in_raw_ptr;
		float* device_out_raw_ptr;
	};
}; // namespace io
}; // namespace rhythm_toolkit

#endif /* RHYTHM_TOOLKIT_IO_H */
