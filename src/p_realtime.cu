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

// real hpss code is below
// the public namespace is to hide cuda details away from the public interface
rhythm_toolkit::hpss::PRealtimeGPU::PRealtimeGPU(float fs,
                                 std::size_t hop,
                                 float beta,
                                 rhythm_toolkit::io::IOGPU& io)
	: io(io)
{
	p_impl = new rhythm_toolkit_private::hpss::PRealtimeGPU(fs, hop, beta);
}

rhythm_toolkit::hpss::PRealtimeGPU::PRealtimeGPU(float fs,
                                 std::size_t hop,
                                 rhythm_toolkit::io::IOGPU& io)
	: PRealtimeGPU(fs, hop, 2.5, io) {};

// best-performing defaults
rhythm_toolkit::hpss::PRealtimeGPU::PRealtimeGPU(float fs, rhythm_toolkit::io::IOGPU& io)
	: PRealtimeGPU(fs, 256, 2.5, io) {};

void rhythm_toolkit::hpss::PRealtimeGPU::process_next_hop()
{
	p_impl->process_next_hop(io.device_in);
	thrust::copy(p_impl->percussive_out.begin(), p_impl->percussive_out.end(), io.device_out);
}

rhythm_toolkit::hpss::PRealtimeGPU::~PRealtimeGPU() {
	delete p_impl;
}

void rhythm_toolkit_private::hpss::PRealtimeGPU::process_next_hop(thrust::device_ptr<float> in_hop)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
	             percussive_out.begin());
	thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	// append latest hop samples e.g. input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(in_hop, in_hop + hop, input.begin() + hop);

	// populate curr_fft with input .* square root von hann window
	thrust::transform(input.begin(), input.end(), win.window.begin(),
	                  curr_fft.begin(), rhythm_toolkit_private::hpss::window_functor());

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
	                  rhythm_toolkit_private::hpss::complex_abs_functor());

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

	// compute percussive mask from harmonic + percussive magnitude spectra
	// the last column of percussive_matrix contains the mask to be applied to the initial fft
	thrust::transform(percussive_matrix.end() - nfft, percussive_matrix.end(),
	                  harmonic_matrix.end() - nfft, percussive_matrix.begin(),
	                  rhythm_toolkit_private::hpss::mask_functor(beta));

	// apply last column of percussive mask to recover percussive audio from original fft
	thrust::transform(curr_fft.begin(), curr_fft.end(),
	                  percussive_matrix.end()-nfft, curr_fft.begin(),
	                  rhythm_toolkit_private::hpss::apply_mask_functor());
	
	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
	                  percussive_out.begin(), percussive_out.begin(),
	                  rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));
}
