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
	class IOGPU {
	public:
		float* host_in;
		float* host_out;
		thrust::device_ptr<float> device_in;
		thrust::device_ptr<float> device_out;
		std::size_t size;

		IOGPU(std::size_t size) : size(size)
		{
			cudaSetDeviceFlags(cudaDeviceMapHost);

			cudaError_t cuda_error;
			unsigned int cuda_malloc_flags
			    = cudaHostAllocMapped | cudaHostAllocPortable;

			// use mapped + wc for performance:
			// http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/online/group__CUDART__MEMORY_g217d441a73d9304c6f0ccc22ec307dba.html
			cuda_error = cudaHostAlloc(
			    ( void** )&host_in, size * sizeof(float),
			    cuda_malloc_flags | cudaHostAllocWriteCombined);

			if (cuda_error != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostAlloc(
			    ( void** )&host_out, size * sizeof(float), cuda_malloc_flags);

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

		~IOGPU()
		{
			cudaFree(host_in);
			cudaFree(host_out);
		}

	private:
		float* device_in_raw_ptr;
		float* device_out_raw_ptr;
	};

	class IOCPU {
	public:
		std::vector<float> host_in;
		std::vector<float> host_out;
		std::size_t size;

		IOCPU(std::size_t size)
		: size(size)
		, host_in(std::vector<float>(size))
		, host_out(std::vector<float>(size)) {};
	};
}; // namespace io
}; // namespace rhythm_toolkit

#endif /* RHYTHM_TOOLKIT_IO_H */
