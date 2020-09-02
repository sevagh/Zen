#include "hpss.h"
#include "rhythm_toolkit/hpss.h"
#include "rhythm_toolkit/rhythm_toolkit.h"
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

typedef thrust::device_vector<thrust::complex<float>>::iterator Iterator;

// real hpss code is below
// the public namespace is to hide cuda details away from the public interface
rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs,
                                                     std::size_t hop_h,
                                                     std::size_t hop_p,
                                                     float beta_h,
                                                     float beta_p)
    : io_h(rhythm_toolkit::io::IOGPU(hop_h))
    , io_p(rhythm_toolkit::io::IOGPU(hop_p))
    , hop_h(hop_h)
    , hop_p(hop_p)
{
	if (hop_h % hop_p != 0) {
		throw rhythm_toolkit::RtkException("hop_h and hop_p should be evenly "
		                                   "divisible");
	}
	h_p_hop_multiplier = hop_h / hop_p;

	p_impl_h
	    = new rhythm_toolkit_private::hpss::HPROfflineGPU(fs, hop_h, beta_h);
	p_impl_p
	    = new rhythm_toolkit_private::hpss::HPROfflineGPU(fs, hop_p, beta_p);
}
rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs,
                                                     std::size_t hop_h,
                                                     std::size_t hop_p)
    : HPRIOfflineGPU(fs, hop_h, hop_p, 2.0, 2.0){};

rhythm_toolkit::hpss::HPRIOfflineGPU::HPRIOfflineGPU(float fs)
    : HPRIOfflineGPU(fs, 4096, 256, 2.0, 2.0){};

std::vector<float>
rhythm_toolkit::hpss::HPRIOfflineGPU::process(std::vector<float> audio)
{
	// loop over whole song here
	int audio_size = audio.size();

	// return same-sized vectors as a result
	int original_size = audio_size;

	int n_chunks_ceil_iter1
	    = ( int )(ceilf(( float )audio_size / ( float )hop_h));
	int pad = n_chunks_ceil_iter1 * hop_h - audio_size;

	int lag1 = p_impl_h->lag;

	// pad with extra lag frames since offline/anticausal HPSS uses future
	// audio to produce past samples
	pad += lag1 * hop_h;

	// first lag frames are simply for prefilling the future frames of the stft
	n_chunks_ceil_iter1 += lag1;

	audio.resize(audio.size() + pad, 0.0F);

	std::vector<float> percussive_out(audio.size());

	// 2nd iteration uses xp1 + xr1 as the total content
	std::vector<float> intermediate(audio.size());

	for (int i = 0; i < n_chunks_ceil_iter1; ++i) {
		// first apply HPR with hop size 4096 for good harmonic separation
		// copy the song, 4096 samples at a time, into the 4096-sized
		// mapped/pinned IOGPU device
		thrust::copy(audio.begin() + i * hop_h,
		             audio.begin() + (i + 1) * hop_h, io_h.host_in);

		// process every 4096-sized hop
		p_impl_h->process_next_hop(io_h.device_in);

		// use xp1 + xr1 as input for the second iteration of HPR with hop size
		// 256 for good percussive separation copy them into separate vectors
		thrust::transform(p_impl_h->percussive_out.begin(),
		                  p_impl_h->percussive_out.begin() + hop_h,
		                  p_impl_h->residual_out.begin(), io_h.device_out,
		                  rhythm_toolkit_private::hpss::sum_vectors_functor());

		std::copy(io_h.host_out, io_h.host_out + hop_h,
		          intermediate.begin() + i * hop_h);
	}

	int n_chunks_ceil_iter2
	    = ( int )(ceilf(( float )original_size / ( float )hop_p));

	int lag2 = p_impl_p->lag;

	// padded with extra lag frames since offline/anticausal HPSS uses future
	// audio to produce past samples
	n_chunks_ceil_iter2 += lag2;

	for (int i = 0; i < n_chunks_ceil_iter2; ++i) {
		// next apply HPR with hop size 256 for good harmonic separation
		// copy the song, 256 samples at a time, into the 256-sized
		// mapped/pinned IOGPU device

		// offset intermediate by lag1*hop_h to skip the padded lag frames of
		// iter1
		std::copy(intermediate.begin() + lag1 * hop_h + i * hop_p,
		          intermediate.begin() + lag1 * hop_h + (i + 1) * hop_p,
		          io_p.host_in);

		p_impl_p->process_next_hop(io_p.device_in, true);

		thrust::copy(p_impl_p->percussive_out.begin(),
		             p_impl_p->percussive_out.begin() + hop_p, io_p.device_out);

		std::copy(io_p.host_out, io_p.host_out + hop_p,
		          percussive_out.begin() + i * hop_p);
	}

	// rotate the useful part by lag2*hop_p to skip the padded lag frames of
	// iter2
	std::copy(percussive_out.begin() + lag2 * hop_p, percussive_out.end(),
	          percussive_out.begin());

	// truncate all padding
	percussive_out.resize(original_size);

	return percussive_out;
}

rhythm_toolkit::hpss::HPRIOfflineGPU::~HPRIOfflineGPU()
{
	delete p_impl_h;
	delete p_impl_p;
}

void rhythm_toolkit_private::hpss::HPROfflineGPU::process_next_hop(
    thrust::device_ptr<float> in_hop,
    bool only_percussive)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
	             percussive_out.begin());
	thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	if (!only_percussive) {
		thrust::copy(harmonic_out.begin() + hop, harmonic_out.end(),
		             harmonic_out.begin());
		thrust::fill(harmonic_out.begin() + hop, harmonic_out.end(), 0.0);

		thrust::copy(residual_out.begin() + hop, residual_out.end(),
		             residual_out.begin());
		thrust::fill(residual_out.begin() + hop, residual_out.end(), 0.0);
	}

	// append latest hop samples e.g. input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(in_hop, in_hop + hop, input.begin() + hop);

	// populate curr_fft with input .* square root von hann window
	thrust::transform(input.begin(), input.end(), win.window.begin(),
	                  curr_fft.begin(),
	                  rhythm_toolkit_private::hpss::window_functor());

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
	time.filter(( Npp32f* )thrust::raw_pointer_cast(s_mag.data()),
	            ( Npp32f* )thrust::raw_pointer_cast(harmonic_matrix.data()));
	frequency.filter(
	    ( Npp32f* )thrust::raw_pointer_cast(s_mag.data()),
	    ( Npp32f* )thrust::raw_pointer_cast(percussive_matrix.data()));

	// compute percussive mask from harmonic + percussive magnitude spectra
	thrust::transform(percussive_matrix.end() - lag * nfft,
	                  percussive_matrix.end() - (lag - 1) * nfft,
	                  harmonic_matrix.end() - lag * nfft,
	                  percussive_mask.begin(),
	                  rhythm_toolkit_private::hpss::mask_functor(beta));

	// apply lag column of percussive mask to recover percussive audio from
	// original fft
	thrust::transform(sliding_stft.end() - lag * nfft,
	                  sliding_stft.end() - (lag - 1) * nfft,
	                  percussive_mask.begin(), curr_fft.begin(),
	                  rhythm_toolkit_private::hpss::apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	// now curr_fft has the current iteration's fresh samples
	// we overlap-add it the real part to the previous
	thrust::transform(
	    curr_fft.begin(), curr_fft.begin() + nwin, percussive_out.begin(),
	    percussive_out.begin(),
	    rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));

	if (only_percussive) {
		return;
	}

	// otherwise proceed to reconstruct harmonic and residuals from masks

	// compute harmonic mask from harmonic + percussive magnitude spectra
	thrust::transform(harmonic_matrix.end() - lag * nfft,
	                  harmonic_matrix.end() - (lag - 1) * nfft,
	                  percussive_matrix.end() - lag * nfft,
	                  harmonic_mask.begin(),
	                  rhythm_toolkit_private::hpss::mask_functor(beta - Eps));

	// compute residual mask from harmonic and percussive masks
	thrust::transform(harmonic_mask.begin(), harmonic_mask.end(),
	                  percussive_mask.begin(), residual_mask.begin(),
	                  rhythm_toolkit_private::hpss::residual_mask_functor());

	// store the harmonic one in its own fft to store for future spectral
	// denoising
	thrust::transform(sliding_stft.end() - lag * nfft,
	                  sliding_stft.end() - (lag - 1) * nfft,
	                  harmonic_mask.begin(), curr_fft.begin(),
	                  rhythm_toolkit_private::hpss::apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	thrust::transform(
	    curr_fft.begin(), curr_fft.begin() + nwin, harmonic_out.begin(),
	    harmonic_out.begin(),
	    rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));

	thrust::transform(sliding_stft.end() - lag * nfft,
	                  sliding_stft.end() - (lag - 1) * nfft,
	                  residual_mask.begin(), curr_fft.begin(),
	                  rhythm_toolkit_private::hpss::apply_mask_functor());

	cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);

	thrust::transform(
	    curr_fft.begin(), curr_fft.begin() + nwin, residual_out.begin(),
	    residual_out.begin(),
	    rhythm_toolkit_private::hpss::overlap_add_functor(COLA_factor));
}
