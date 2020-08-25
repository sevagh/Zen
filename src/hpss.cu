#include "hpss.h"
#include "nppdefs.h"
#include "nppi.h"
#include "rhythm_toolkit/hpss.h"
#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <cufft.h>
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

static constexpr float eps = std::numeric_limits<float>::epsilon();

// real hpss code is below
// the public namespace is to hide cuda details away from the public interface
rhythm_toolkit::hpss::HPSS::HPSS(float fs,
                                 std::size_t hop,
                                 float beta,
                                 rhythm_toolkit::io::IO& io)
{
	p_impl = new rhythm_toolkit_private::hpss::HPSS(fs, hop, beta, io);
}

rhythm_toolkit::hpss::HPSS::HPSS(float fs,
                                 std::size_t hop,
                                 rhythm_toolkit::io::IO& io)
{
	p_impl = new rhythm_toolkit_private::hpss::HPSS(fs, hop, 2.0, io);
}

rhythm_toolkit::hpss::HPSS::HPSS(float fs, rhythm_toolkit::io::IO& io)
{
	p_impl = new rhythm_toolkit_private::hpss::HPSS(fs, 512, 2.0, io);
}

void rhythm_toolkit::hpss::HPSS::process_next_hop()
{
	p_impl->process_next_hop();
}

void rhythm_toolkit::hpss::HPSS::peek_separated_percussive()
{
	p_impl->peek_separated_percussive();
}

void rhythm_toolkit::hpss::HPSS::reset() { p_impl->reset(); }

rhythm_toolkit::hpss::HPSS::~HPSS() { delete p_impl; }

struct window_functor {
	window_functor() {}

	__host__ __device__ thrust::complex<float> operator()(const float& x,
	                                                      const float& y) const
	{
		return thrust::complex<float>{x * y, 0.0};
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
	__host__ __device__ ValueType operator()(const thrust::complex<ValueType>& z)
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

	__host__ __device__ float operator()(const float& x, const float& y) const
	{
		return float((x / (y + eps)) >= beta);
	}
};

void rhythm_toolkit_private::hpss::HPSS::process_next_hop()
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
	             percussive_out.begin());
	thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	thrust::copy(
	    harmonic_out.begin() + hop, harmonic_out.end(), harmonic_out.begin());
	thrust::fill(harmonic_out.begin() + hop, harmonic_out.end(), 0.0);

	// append latest hop samples e.g. input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(io.device_in, io.device_in + hop, input.begin() + hop);

	// populate curr_fft with input .* square root von hann window
	thrust::transform(input.begin(), input.end(), win.window.begin(),
	                  curr_fft.begin(), window_functor());

	// zero out the second half of the fft
	thrust::fill(curr_fft.begin() + nwin, curr_fft.end(),
	             thrust::complex<float>{0.0, 0.0});
	cufftExecC2C(plan_forward, fft_ptr, fft_ptr, CUFFT_FORWARD);

	// rotate stft matrix to move the oldest column to the end
	// copy curr_fft into the last column of the stft
	thrust::copy(
	    sliding_stft.begin() + nfft, sliding_stft.end(), sliding_stft.begin());
	thrust::copy(curr_fft.begin(), curr_fft.end(), sliding_stft.end() - nfft);

	// calculate the magnitude of the stft
	thrust::transform(sliding_stft.begin(), sliding_stft.end(), s_mag.begin(),
	                  complex_abs_functor());

	// apply median filter in horizontal and vertical directions with NPP
	nppiFilterMedian_32f_C1R(thrust::raw_pointer_cast(s_mag.data()), nstep,
	                         thrust::raw_pointer_cast(harmonic_matrix.data()),
	                         nstep, medfilt_roi, harmonic_filter_mask,
	                         harmonic_anchor, harmonic_buffer);
	nppiFilterMedian_32f_C1R(
	    thrust::raw_pointer_cast(s_mag.data()), nstep,
	    thrust::raw_pointer_cast(percussive_matrix.data()), nstep, medfilt_roi,
	    percussive_filter_mask, percussive_anchor, percussive_buffer);

	thrust::transform(harmonic_matrix.end() - nfft, harmonic_matrix.end(),
	                  percussive_matrix.end() - nfft, harmonic_mask.begin(),
	                  mask_functor(beta - eps));
	thrust::transform(percussive_matrix.end() - nfft, percussive_matrix.end(),
	                  harmonic_matrix.end() - nfft, percussive_mask.begin(),
	                  mask_functor(beta));

	// apply last column of percussive mask to recover separated fft
	thrust::transform(sliding_stft.end() - nfft, sliding_stft.end(),
	                  percussive_mask.begin(), curr_fft.begin(),
	                  apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  percussive_out.begin(), percussive_out.begin(),
	                  overlap_add_functor(COLA_factor));

	// apply last column of percussive mask to recover separated fft
	thrust::transform(sliding_stft.end() - nfft, sliding_stft.end(),
	                  harmonic_mask.begin(), curr_fft.begin(),
	                  apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now harm_fft has the current iteration's fresh samples
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  harmonic_out.begin(), harmonic_out.begin(),
	                  overlap_add_functor(COLA_factor));
}
