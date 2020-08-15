#include "hpss.h"
#include "nppdefs.h"
#include "nppi.h"
#include "rhythm_toolkit/hpss.h"
#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

// real hpss code is below
// the public namespace is to hide cuda details away from the public interface
rhythm_toolkit::hpss::HPSS::HPSS(float fs, std::size_t hop, float beta)
{
	p_impl = new rhythm_toolkit_private::hpss::HPSS(fs, hop, beta);
}

rhythm_toolkit::hpss::HPSS::HPSS(float fs)
{
	p_impl = new rhythm_toolkit_private::hpss::HPSS(fs, 512, 2.0);
}

void rhythm_toolkit::hpss::HPSS::process_next_hop(std::vector<float>& next_hop)
{
	p_impl->process_next_hop(next_hop);
}

std::vector<float> rhythm_toolkit::hpss::HPSS::peek_separated_percussive()
{
	return p_impl->peek_separated_percussive();
}

rhythm_toolkit::hpss::HPSS::~HPSS() { delete p_impl; }

struct window_functor {
	window_functor() {}

	__host__ __device__ float operator()(const float& x, const float& y) const
	{
		return x * y;
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

	__host__ __device__ float operator()(const float& x, const float& y) const
	{
		return x + y * cola_factor;
	}
};

struct complex_abs_functor {
	template <typename ValueType>
	__host__ __device__ ValueType operator()(const thrust::complex<ValueType>& z)
	{
		return thrust::abs(z);
	}
};

struct mask_functor {
	const float beta;
	static constexpr float eps = std::numeric_limits<float>::epsilon();

	mask_functor(float _beta)
	    : beta(_beta)
	{
	}

	__host__ __device__ float operator()(const float& x, const float& y) const
	{
		return float(x / (y + eps) >= beta);
	}
};

void rhythm_toolkit_private::hpss::HPSS::process_next_hop(
    std::vector<float>& current_hop)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
	             percussive_out.begin());
	thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	// append latest hop samples e.g.
	//     input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(current_hop.begin(), current_hop.end(), input.begin() + hop);

	// apply square root von hann window to new samples input[hop:]
	thrust::transform(input.begin() + hop, input.end(), win.window.begin(),
	                  input.begin(), window_functor());

	cufftExecR2C(plan_forward, in_ptr, fft_ptr);

	// rotate stft matrix to move the oldest column to the end
	// copy curr_fft into the last column of the stft
	thrust::copy(
	    sliding_stft.begin() + nfft, sliding_stft.end(), sliding_stft.begin());
	thrust::copy(curr_fft.begin(), curr_fft.end(),
	             sliding_stft.begin() + (stft_width - 1) * nfft);

	// calculate the magnitude of the stft
	thrust::transform(sliding_stft.begin(), sliding_stft.end(), s_mag.begin(),
	                  complex_abs_functor());

	// apply median filter in horizontal and vertical directions with NPP
	nppiFilterMedian_32f_C1R(thrust::raw_pointer_cast(s_mag.data()), nfft,
	                         thrust::raw_pointer_cast(harmonic_matrix.data()),
	                         nfft, harmonic_roi, harmonic_mask,
	                         harmonic_anchor, harmonic_buffer);
	nppiFilterMedian_32f_C1R(
	    thrust::raw_pointer_cast(s_mag.data()), nfft,
	    thrust::raw_pointer_cast(percussive_matrix.data()), nfft,
	    percussive_roi, percussive_mask, percussive_anchor, percussive_buffer);

	// calculate the binary masks in-place
	//
	// note that from this point on, we only consider the percussive part of
	// the algorithm because the horizontal median filter works poorly
	// in real-time overwrite the matrices in-place
	thrust::transform(percussive_matrix.begin(), percussive_matrix.end(),
	                  harmonic_matrix.begin(), percussive_matrix.begin(),
	                  mask_functor(beta));

	// apply last column of percussive mask to recover separated fft
	// recycle curr_fft instead of allocating a new vector
	thrust::transform(curr_fft.begin(), curr_fft.end(),
	                  percussive_matrix.end() - nfft, curr_fft.begin(),
	                  apply_mask_functor());

	cufftExecC2R(plan_backward, fft_ptr, out_ptr);

	// now percussive_out_tmp has the current iteration's fresh samples
	// we overlap-add it to the previous
	thrust::transform(percussive_out_tmp.begin(), percussive_out_tmp.end(),
	                  percussive_out.begin(), percussive_out.begin(),
	                  overlap_add_functor(COLA_factor));

	for (int i = 0; i < 8; ++i) {
		std::cout << percussive_out[i] << " " << std::endl;
	}
	std::cout << std::endl;
}
