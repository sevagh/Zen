#ifndef HPSS_COMMON_PRIVATE_H
#define HPSS_COMMON_PRIVATE_H

#include "medianfilter.h"
#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rhythm_toolkit_private {
namespace hpss {
	static constexpr float Eps = std::numeric_limits<float>::epsilon();

	// various thrust GPU functors, reused across HPRIOfflineGPU and PRealtimeGPU
	struct window_functor {
		window_functor() {}

		__host__ __device__ thrust::complex<float>
		operator()(const float& x, const float& y) const
		{
			return thrust::complex<float>{x * y, 0.0};
		}
	};

	struct residual_mask_functor {
		residual_mask_functor() {}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return 1 - (x + y);
		}
	};

	struct apply_mask_functor {
		apply_mask_functor() {}

		__host__ __device__ thrust::complex<float>
		operator()(const thrust::complex<float>& x, const float& y) const
		{
			return x * y;
		}
	};

	struct overlap_add_functor {
		const float cola_factor;
		overlap_add_functor(float _cola_factor)
		    : cola_factor(_cola_factor)
		{
		}

		__host__ __device__ float operator()(const thrust::complex<float>& x,
		                                     const float& y) const
		{
			return y + x.real() * cola_factor;
		}
	};

	struct complex_abs_functor {
		template <typename ValueType>
		__host__ __device__ ValueType
		operator()(const thrust::complex<ValueType>& z)
		{
			return thrust::abs(z);
		}
	};

	struct mask_functor {
		const float beta;

		mask_functor(float _beta)
		    : beta(_beta)
		{
		}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return float((x / (y + Eps)) >= beta);
		}
	};

	struct sum_vectors_functor {
		sum_vectors_functor() {}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return x + y;
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* HPSS_COMMON_PRIVATE_H */
