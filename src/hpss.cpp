#include "rhythm_toolkit/hpss.h"

// uses a locally vendored copy of
// https://github.com/suomela/mf2d/blob/master/src/filter.h for median filtering
#include "filter.h"
#include <ffts/ffts.h>
#include <math.h>

void rhythm_toolkit::hpss::HPSS::process_next_hop(std::vector<float>& current_hop)
{
	// append latest hop samples e.g.
	//     input = input[hop:] + current_hop
	std::rotate(input.begin(), input.begin() + hop, input.end());
	std::copy(current_hop.begin(), current_hop.end(), input.begin() + hop);

	// apply square root von hann window to input
	for (std::size_t i = 0; i < nwin; ++i) {
		input_windowed[i] = input[i] * win.window[i];
	}

	// do the real forward fft in-place
	std::transform(input_windowed.begin(), input_windowed.end(),
	               curr_fft.begin(), [](float x) -> std::complex<float> {
		               return std::complex<float>(x, 0.0f);
	               });
	ffts_execute(fft_forward, curr_fft.data(), curr_fft.data());

	// rotate stft matrix to move the oldest column to the end
	// copy curr_fft into the last column of the stft
	std::rotate(
	    sliding_stft.begin(), sliding_stft.begin() + nfft, sliding_stft.end());
	std::copy(curr_fft.begin(), curr_fft.end(),
	          sliding_stft.begin() + (stft_width - 1) * nfft);

	// calculate half the magnitude of the stft
	// TODO: pragma omp stuff here? parallelize these loops
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin; ++j) {
			// |complex| = sqrt(r*r + i*i)
			s_half_mag[i * nwin + j] = std::abs(sliding_stft[i * nfft + j]);
		}
	}

	/******************************
	 * SO FAR IDENTICAL TO MATLAB *
	 ******************************/

	// apply median filter in horizontal and vertical directions
	// only consider half the stft
	median_filter_2d<float>(( int )nfft, ( int )stft_width, 0, l_harm, 0,
	                        s_half_mag.data(), harmonic_matrix.data());
	median_filter_2d<float>(( int )nfft, ( int )stft_width, l_perc, 0, 0,
	                        s_half_mag.data(), percussive_matrix.data());

	// calculate the binary masks
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin + 1; ++j) {
			auto idx = i * (nwin + 1) + j;
			harmonic_mask[idx]
			    = float(std::real(harmonic_matrix[idx])
			                / (std::real(percussive_matrix[idx]) + eps)
			            > beta);
			percussive_mask[idx]
			    = float(std::real(percussive_matrix[idx])
			                / (std::real(harmonic_matrix[idx]) + eps)
			            >= beta);
		}
	}

	std::complex<float> scale = {1.0f / ( float )nfft, 0.0};

	// apply masks to recover separated fft
	std::size_t inverse_i = nfft - 1;
	for (std::size_t i = 0; i < nwin; ++i) {
		auto mask_idx = (stft_width - 1) * nwin + i;
		harmonic_fft[i] = curr_fft[i] * harmonic_mask[mask_idx] * scale;

		// H = H + flipud(conj(H))
		harmonic_fft[inverse_i - i] = std::conj(harmonic_fft[i]);

		percussive_fft[i] = curr_fft[i] * percussive_mask[mask_idx] * scale;
		percussive_fft[inverse_i - i] = std::conj(percussive_fft[i]);
	}

	// apply ifft to get resultant audio in-place
	// take the real part into the out arrays
	ffts_execute(fft_backward, harmonic_fft.data(), harmonic_fft.data());
	ffts_execute(fft_backward, percussive_fft.data(), percussive_fft.data());
	std::transform(
	    harmonic_fft.begin(), harmonic_fft.end(), harmonic_out_raw.begin(),
	    [](std::complex<float> x) -> float { return std::real(x); });
	std::transform(percussive_fft.begin(), percussive_fft.end(),
	               percussive_out_raw.begin(),
	               [](std::complex<float> x) -> float { return std::real(x); });

	// weighted overlap add with last iteration's samples - only half of the
	// real fft matters cola divide factor is for COLA compliance see
	// https://github.com/sevagh/Real-Time-HPSS for background
	for (std::size_t i = 0; i < nwin; ++i) {
		harmonic_out[i] += harmonic_out_raw[i] * COLA_factor;
		percussive_out[i] += percussive_out_raw[i] * COLA_factor;

		// after weighted overlap add, this is the data we're ready to return
		if (i < hop) {
			harmonic_out_hop[i] = harmonic_out[i];
			percussive_out_hop[i] = percussive_out[i];
		}
	}

	// finally we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	std::rotate(
	    harmonic_out.begin(), harmonic_out.begin() + hop, harmonic_out.end());
	std::rotate(percussive_out.begin(), percussive_out.begin() + hop,
	            percussive_out.end());
	std::fill(harmonic_out.begin() + hop, harmonic_out.end(), 0.0);
	std::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);
}
