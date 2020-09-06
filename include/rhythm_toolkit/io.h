#ifndef RHYTHM_TOOLKIT_IO_H
#define RHYTHM_TOOLKIT_IO_H

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <numeric>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "npp.h"

#include "rhythm_toolkit.h"

static std::once_flag cuda_init_flag;

static void cuda_init()
{
	std::call_once(
	    cuda_init_flag, []() { cudaSetDeviceFlags(cudaDeviceMapHost); });
}

namespace rhythm_toolkit {
namespace io {
	class IOGPU {
	public:
		float* host_in;
		float* host_out;
		thrust::device_ptr<float> device_in;
		thrust::device_ptr<float> device_out;
		std::size_t size;

		IOGPU(std::size_t size)
		    : size(size)
		{
			cuda_init();

			cudaError_t cuda_error;
			unsigned int cuda_malloc_flags
			    = cudaHostAllocMapped | cudaHostAllocPortable;

			// use mapped + wc for performance:
			// http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/online/group__CUDART__MEMORY_g217d441a73d9304c6f0ccc22ec307dba.html
			cuda_error = cudaHostAlloc(
			    ( void** )&host_in, size * sizeof(float),
			    cuda_malloc_flags | cudaHostAllocWriteCombined);

			if (cuda_error != cudaSuccess) {
				std::cerr << "IOGPU: cuda malloc error: "
				          << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostAlloc(
			    ( void** )&host_out, size * sizeof(float), cuda_malloc_flags);

			if (cuda_error != cudaSuccess) {
				std::cerr << "IOGPU: cuda malloc error: "
				          << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostGetDevicePointer(
			    ( void** )(&device_in_raw_ptr), ( void* )host_in, 0);
			if (cuda_error != cudaSuccess) {
				std::cerr << "IOGPU: cuda devptr error: "
				          << cudaGetErrorString(cuda_error) << std::endl;
				std::exit(-1);
			}

			cuda_error = cudaHostGetDevicePointer(
			    ( void** )(&device_out_raw_ptr), ( void* )host_out, 0);
			if (cuda_error != cudaSuccess) {
				std::cerr << "IOGPU: cuda devptr error: "
				          << cudaGetErrorString(cuda_error) << std::endl;
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
}; // namespace io
}; // namespace rhythm_toolkit

#endif /* RHYTHM_TOOLKIT_IO_H */
