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
rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs,
                                 std::size_t max_size_samples,
                                 std::size_t hop_h,
                                 std::size_t hop_p,
                                 float beta_h,
                                 float beta_p,
                                 rhythm_toolkit::io::IOGPU& io)
	: io(io)
	, intermediate(thrust::device_vector<float>(max_size_samples))
	, hop_h(hop_h)
	, hop_p(hop_p)
{
	std::cout << "initial params, max: " << max_size_samples << ", hop_h: " << hop_h << ", hop_p: " << hop_p << std::endl;
	p_impl_h = new rhythm_toolkit_private::hpss::HPROfflineGPU(fs, max_size_samples, hop_h, beta_h);
	p_impl_p = new rhythm_toolkit_private::hpss::HPROfflineGPU(fs, max_size_samples, hop_p, beta_p);
}

rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs,
                                 std::size_t max_size_samples,
                                 std::size_t hop_h,
                                 std::size_t hop_p,
                                 rhythm_toolkit::io::IOGPU& io)
	: HPRIOfflineGPU(fs, max_size_samples, hop_h, hop_p, 2.0, 2.0, io) {};

rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs, std::size_t max_size_samples, rhythm_toolkit::io::IOGPU& io)
	: HPRIOfflineGPU(fs, max_size_samples, 4096, 256, 2.0, 2.0, io) {};

void rhythm_toolkit::hpss::HPRIOfflineGPU::process()
{
	std::cout << "A" << std::endl;

	// first apply HPR with hop size 4096 for good harmonic separation
	p_impl_h->process(io.device_in, io.size);

	std::cout << "B" << std::endl;

	// use xp1 + xr1 as input for the second iteration of HPR with hop size 256 for good percussive separation
	thrust::transform(p_impl_h->percussive_out.begin(), p_impl_h->percussive_out.end(), p_impl_h->residual_out.begin(), intermediate.begin(), rhythm_toolkit_private::hpss::sum_vectors_functor());

	std::cout << "C" << std::endl;

	// do second pass with input signal = xp1 + xr1
	p_impl_p->process(intermediate.data(), io.size);

	// populate final output vector with xp2, the second stage percussive separation
	thrust::copy(p_impl_p->percussive_out.begin(), p_impl_p->percussive_out.end(), io.device_out);
}

rhythm_toolkit::hpss::HPRIOfflineGPU::~HPRIOfflineGPU() {
	delete p_impl_h;
	delete p_impl_p;
}

void rhythm_toolkit_private::hpss::HPROfflineGPU::process(thrust::device_ptr<float> in_signal, std::size_t signal_size)
{
	std::size_t n_chunks = (std::size_t)ceilf((float)signal_size/(float)hop);
	std::size_t remain = signal_size - (n_chunks-1)*hop;

	std::cout <<  "prefill stft" << std::endl;

	std::cout << "n chunks: " << n_chunks << std::endl;
	std::cout << "stft width: " << stft_width << std::endl;
	std::cout << "signal_size/hop: " << (float)signal_size/(float)hop << std::endl;
	std::cout << "remain: " << remain << std::endl;

	// pre-fill the entire stft
	for (std::size_t i = 0; i < n_chunks-1; ++i) {
		// copy current hop samples e.g. input = input[hop:] + current_hop
		thrust::copy(input.begin() + hop, input.end(), input.begin());
		thrust::copy(in_signal + i*hop, in_signal + (i+1)*hop, input.begin() + hop);

		// populate curr_fft with input .* square root von hann window
		thrust::transform(input.begin(), input.end(), win.window.begin(),
				  curr_fft.begin(), rhythm_toolkit_private::hpss::window_functor());

		// zero out the second half of the fft
		thrust::fill(curr_fft.begin() + nwin, curr_fft.end(),
			     thrust::complex<float>{0.0, 0.0});

		// in the offline variant, we need future stft colums available
		cufftExecC2C(plan_forward, fft_ptr, fft_ptr, CUFFT_FORWARD);

		// insert fft into stft
		thrust::copy(curr_fft.begin(), curr_fft.end(), sliding_stft.begin() + i*nfft);
	}

	// for remaining samples
	for (std::size_t i = n_chunks-2; i < n_chunks; ++i) {
		// copy current hop samples e.g. input = input[hop:] + current_hop
		thrust::copy(input.begin() + hop, input.end(), input.begin());
		thrust::copy(in_signal + i*hop, in_signal + i*hop + remain, input.begin() + hop);

		// zero out anything past remain
		thrust::fill(input.begin()+hop+remain, input.end(), 0.0F);

		// populate curr_fft with input .* square root von hann window
		thrust::transform(input.begin(), input.end(), win.window.begin(),
				  curr_fft.begin(), rhythm_toolkit_private::hpss::window_functor());

		// zero out the second half of the fft
		thrust::fill(curr_fft.begin() + nwin, curr_fft.end(),
			     thrust::complex<float>{0.0, 0.0});

		// in the offline variant, we need future stft colums available
		cufftExecC2C(plan_forward, fft_ptr, fft_ptr, CUFFT_FORWARD);

		// insert fft into stft
		thrust::copy(curr_fft.begin(), curr_fft.end(), sliding_stft.begin() + i*nfft);
	}

	std::cout <<  "calculate median filters & masks 1" << std::endl;

	// calculate the magnitude of the stft
	thrust::transform(sliding_stft.begin(), sliding_stft.end(), s_mag.begin(),
			  rhythm_toolkit_private::hpss::complex_abs_functor());

	std::cout <<  "calculate median filters & masks 2" << std::endl;

	// apply median filter in horizontal and vertical directions with NPP
	// to create percussive and harmonic spectra
	nppiFilterMedian_32f_C1R(thrust::raw_pointer_cast(s_mag.data()) + harmonic_start_pixel_offset, harmonic_nstep,
				 thrust::raw_pointer_cast(harmonic_matrix.data()) + harmonic_start_pixel_offset,
				 harmonic_nstep, harmonic_roi, harmonic_filter_mask,
				 harmonic_anchor, harmonic_buffer);

	std::cout <<  "calculate median filters & masks 3" << std::endl;

	std::cout <<  "s_mag.size: " << s_mag.size() << std::endl;
	std::cout <<  "percussive_matrix.size: " << percussive_matrix.size() << std::endl;
	std::cout <<  "harmonic.roi: " << harmonic_roi.width << ", " << harmonic_roi.height << std::endl;
	std::cout <<  "perc mask: " << percussive_filter_mask.width << ", " << percussive_filter_mask.height << std::endl;
	std::cout <<  "perc anchor: " << percussive_anchor.x << ", " << percussive_anchor.y << std::endl;

	std::cout << "s mag ptr: " << thrust::raw_pointer_cast(s_mag.data()) << std::endl;
	std::cout << "p mat ptr: " << thrust::raw_pointer_cast(percussive_matrix.data()) << std::endl;
	std::cout << "p buf ptr: " << (void*)percussive_buffer << std::endl;

	nppiFilterMedian_32f_C1R(
	    thrust::raw_pointer_cast(s_mag.data()) + percussive_start_pixel_offset, percussive_nstep,
	    thrust::raw_pointer_cast(percussive_matrix.data()) + percussive_start_pixel_offset, percussive_nstep, percussive_roi,
	    percussive_filter_mask, percussive_anchor, percussive_buffer);

	std::cout <<  "calculate median filters & masks 4" << std::endl;

	// compute masks
	thrust::transform(percussive_matrix.begin(), percussive_matrix.end(),
			  harmonic_matrix.begin(), percussive_mask.begin(),
			  rhythm_toolkit_private::hpss::mask_functor(beta));

	std::cout <<  "calculate median filters & masks 5" << std::endl;

	thrust::transform(harmonic_matrix.begin(), harmonic_matrix.end(),
			  percussive_matrix.begin(), harmonic_mask.begin(),
			  rhythm_toolkit_private::hpss::mask_functor(beta-rhythm_toolkit_private::hpss::Eps));

	std::cout <<  "calculate median filters & masks 6" << std::endl;

	// compute residual mask from harmonic and percussive masks
	thrust::transform(harmonic_mask.begin(), harmonic_mask.end(),
			  percussive_mask.begin(), residual_mask.begin(),
			  rhythm_toolkit_private::hpss::residual_mask_functor());

	std::cout <<  "sliding overlap add reconstruction" << std::endl;

	for (std::size_t i = 0; i < n_chunks; ++i) {
		// following the previous iteration
		// we rotate the percussive and harmonic arrays to get them ready
		// for the next hop and next overlap add
		thrust::copy(percussive_out.begin() + (i+1)*hop, percussive_out.begin() + (i+1)*nwin,
			     percussive_out.begin() + i*hop);
		thrust::fill(percussive_out.begin() + (i+1)*hop, percussive_out.begin() + (i+1)*nwin, 0.0);

		thrust::copy(harmonic_out.begin() + (i+1)*hop, harmonic_out.begin() + (i+1)*nwin,
			     harmonic_out.begin() + i*hop);
		thrust::fill(harmonic_out.begin() + (i+1)*hop, harmonic_out.begin() + (i+1)*nwin, 0.0);

		thrust::copy(residual_out.begin() + (i+1)*hop, residual_out.begin() + (i+1)*nwin,
			     residual_out.begin() + i*hop);
		thrust::fill(residual_out.begin() + (i+1)*hop, residual_out.begin() + (i+1)*nwin, 0.0);

		// apply current column of percussive mask to recover percussive audio from original fft
		thrust::transform(sliding_stft.begin() + i*nfft, sliding_stft.begin() + (i+1)*nfft,
				  percussive_mask.begin() + i*nfft, curr_fft.begin(),
				  rhythm_toolkit_private::hpss::apply_mask_functor());

		cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

		// now curr_fft has the current iteration's fresh samples
		// we overlap-add it the real part to the previous
		thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
				  percussive_out.begin() + i*hop, percussive_out.begin() + i*hop,
				  rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));

		// apply last column of harmonic mask to recover harmonic audio from original fft
		thrust::transform(sliding_stft.begin() + i*nfft, sliding_stft.begin() + (i+1)*nfft,
				  harmonic_mask.begin() + i*nfft, curr_fft.begin(),
				  rhythm_toolkit_private::hpss::apply_mask_functor());

		cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

		// now curr_fft has the current iteration's fresh samples
		// we overlap-add it the real part to the previous
		thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
				  harmonic_out.begin() + i*hop, harmonic_out.begin() + i*hop,
				  rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));

		// apply last column of residual mask to recover residual audio from original fft
		thrust::transform(sliding_stft.begin() + i*nfft, sliding_stft.begin() + (i+1)*nfft,
				  residual_mask.begin() + i*nfft, curr_fft.begin(),
				  rhythm_toolkit_private::hpss::apply_mask_functor());

		cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

		// now curr_fft has the current iteration's fresh samples
		// we overlap-add it the real part to the previous
		thrust::transform(curr_fft.begin(), curr_fft.begin() + nwin,
				  residual_out.begin() + i*hop, residual_out.begin() + i*hop,
				  rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));
	}
}
