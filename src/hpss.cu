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

struct sum_vectors_functor {
	sum_vectors_functor() {}

	__host__ __device__ float
	operator()(const float& x, const float& y) const
	{
		return x + y;
	}
};

// real hpss code is below
// the public namespace is to hide cuda details away from the public interface
rhythm_toolkit::hpss::HPSS::HPSS(float fs,
                                 std::size_t hop_h,
                                 std::size_t hop_p,
                                 float beta_h,
                                 float beta_p,
                                 rhythm_toolkit::io::IO& io)
	: io(io)
	, intermediate(thrust::device_vector<float>(hop_h))
	, hop_h(hop_h)
	, hop_p(hop_p)
{
	p_impl_h = new rhythm_toolkit_private::hpss::HPSS(fs, hop_h, beta_h);
	p_impl_p = new rhythm_toolkit_private::hpss::HPSS(fs, hop_p, beta_p);
}

rhythm_toolkit::hpss::HPSS::HPSS(float fs,
                                 std::size_t hop_h,
                                 std::size_t hop_p,
                                 rhythm_toolkit::io::IO& io)
	: HPSS(fs, hop_h, hop_p, 2.0, 2.0, io) {};

rhythm_toolkit::hpss::HPSS::HPSS(float fs, rhythm_toolkit::io::IO& io)
	: HPSS(fs, 4096, 256, 2.0, 2.0, io) {};

void rhythm_toolkit::hpss::HPSS::process_next_hop()
{
	// first apply HPR with hop size 4096 for good harmonic separation
	p_impl_h->process_next_hop(io.device_in);
	thrust::copy(p_impl_h->percussive_out.begin(), p_impl_h->percussive_out.end(), io.device_out);

	// use xp1 + xr1 as input for the second iteration of HPR with hop size 256 for good percussive separation
	//thrust::transform(p_impl_h->percussive_out.begin(), p_impl_h->percussive_out.end(), p_impl_h->residual_out.begin(), intermediate.begin(), sum_vectors_functor());

	////thrust::copy(p_impl_h->percussive_out.begin(), p_impl_h->percussive_out.end(), intermediate.begin());

	//for (std::size_t i = 0; i < (std::size_t)((float)hop_h/(float)hop_p); ++i) {
	//	// chunk through the 4096 vector in increments of 256
	//	p_impl_p->process_next_hop(intermediate.data() + i*hop_p);

	//	// populate output vector with the consecutive 256-sized percussive separations
	//	thrust::copy(p_impl_p->percussive_out.begin(), p_impl_p->percussive_out.end(), io.device_out + i*hop_p);
	//}
}

rhythm_toolkit::hpss::HPSS::~HPSS() {
	delete p_impl_h;
	delete p_impl_p;
}

struct window_functor {
	window_functor() {}

	__host__ __device__ thrust::complex<float> operator()(const float& x,
	                                                      const float& y) const
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

void rhythm_toolkit_private::hpss::HPSS::process_next_hop(thrust::device_ptr<float> in_hop)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
	             percussive_out.begin());
	thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	thrust::copy(harmonic_out.begin() + hop, harmonic_out.end(),
	             harmonic_out.begin());
	thrust::fill(harmonic_out.begin() + hop, harmonic_out.end(), 0.0);

	thrust::copy(residual_out.begin() + hop, residual_out.end(),
	             residual_out.begin());
	thrust::fill(residual_out.begin() + hop, residual_out.end(), 0.0);

	// append latest hop samples e.g. input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(in_hop, in_hop + hop, input.begin() + hop);

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
	// to create percussive and harmonic spectra
	nppiFilterMedian_32f_C1R(thrust::raw_pointer_cast(s_mag.data()), nstep,
	                         thrust::raw_pointer_cast(harmonic_matrix.data()),
	                         nstep, medfilt_roi, harmonic_filter_mask,
	                         harmonic_anchor, harmonic_buffer);

	nppiFilterMedian_32f_C1R(
	    thrust::raw_pointer_cast(s_mag.data()), nstep,
	    thrust::raw_pointer_cast(percussive_matrix.data()), nstep, medfilt_roi,
	    percussive_filter_mask, percussive_anchor, percussive_buffer);

	// 3-point median filter to smooth any glitches from the percussion
	//nppiFilterMedian_32f_C1R(
	//    thrust::raw_pointer_cast(percussive_matrix.data()), nstep,
	//    thrust::raw_pointer_cast(percussive_matrix.data()), nstep, medfilt_roi,
	//    percussive_smoothing_mask, percussive_anchor, percussive_smoothing_buffer);

	// compute percussive mask from percussive matrix
	thrust::transform(percussive_matrix.end() - nfft, percussive_matrix.end(),
	                  harmonic_matrix.end() - nfft, percussive_mask.begin(),
	                  mask_functor(beta));

	// apply last column of percussive mask to recover percussive audio from original fft
	thrust::transform(sliding_stft.end()-nfft, sliding_stft.end(),
	                  percussive_mask.begin(), curr_fft.begin(),
	                  apply_mask_functor());
	
	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  percussive_out.begin(), percussive_out.begin(),
	                  overlap_add_functor(COLA_factor));

	// compute harmonic mask from harmonic matrix
	thrust::transform(harmonic_matrix.end() - nfft, harmonic_matrix.end(),
	                  percussive_matrix.end() - nfft, harmonic_mask.begin(),
	                  mask_functor(beta-eps));

	// apply last column of harmonic mask to recover harmonic audio from original fft
	thrust::transform(sliding_stft.end()-nfft, sliding_stft.end(),
	                  harmonic_mask.begin(), curr_fft.begin(),
	                  apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  harmonic_out.begin(), harmonic_out.begin(),
	                  overlap_add_functor(COLA_factor));

	// compute residual mask from harmonic and percussive masks
	thrust::transform(harmonic_mask.begin(), harmonic_mask.end(),
	                  percussive_mask.begin(), residual_mask.begin(),
	                  residual_mask_functor());

	// apply last column of residual mask to recover residual audio from original fft
	thrust::transform(sliding_stft.end()-nfft, sliding_stft.end(),
	                  residual_mask.begin(), curr_fft.begin(),
	                  apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  residual_out.begin(), residual_out.begin(),
	                  overlap_add_functor(COLA_factor));
}
