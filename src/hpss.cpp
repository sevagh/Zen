#include "rhythm_toolkit/hpss.h"

// uses a locally vendored copy of
// https://github.com/suomela/mf2d/blob/master/src/filter.h for median filtering
#include "filter.h"
#include <ffts/ffts.h>
#include <math.h>

void rhythm_toolkit::hpss::HPSS::process_next_hop(std::vector<float>& current_hop)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	std::rotate(percussive_out.begin(), percussive_out.begin() + hop,
	            percussive_out.end());
	std::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);

	// append latest hop samples e.g.
	//     input = input[hop:] + current_hop
	std::rotate(input.begin(), input.begin() + hop, input.end());
	std::copy(current_hop.begin(), current_hop.end(), input.begin() + hop);

	// apply square root von hann window to the newest appended hop samples
	for (std::size_t i = 0; i < nwin; ++i) {
		curr_fft[i] = {input[i] * win.window[i], 0.0F};
	}
	std::fill(curr_fft.begin() + nwin, curr_fft.end(), 0.0F);

	ffts_execute(fft_forward, curr_fft.data(), curr_fft.data());

	// rotate stft matrix to move the oldest column to the end
	// copy curr_fft into the last column of the stft
	std::rotate(
	    sliding_stft.begin(), sliding_stft.begin() + nfft, sliding_stft.end());
	std::copy(curr_fft.begin(), curr_fft.end(),
	          sliding_stft.begin() + (stft_width - 1) * nfft);

	// calculate half the magnitude of the stft
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin; ++j) {
			s_half_mag[i * nwin + j] = std::abs(sliding_stft[i * nfft + j]);
		}
	}

	// apply median filter in horizontal and vertical directions
	// only consider half the stft
	median_filter_2d<float>(( int )nwin, ( int )stft_width, 0, l_harm, 0,
	                        s_half_mag.data(), harmonic_matrix.data());
	median_filter_2d<float>(( int )nwin, ( int )stft_width, l_perc, 0, 0,
	                        s_half_mag.data(), percussive_matrix.data());

	// calculate the binary masks
	// note that from this point on, we only consider the percussive part of
	// the algorithm that's because the horizontal median filter works poorly
	// in real-time overwrite the matrices in-place
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin; ++j) {
			auto idx = i * nwin + j;

			// Mp = P/(H + eps) >= beta
			percussive_matrix[idx]
			    = float(percussive_matrix[idx]
			                / (harmonic_matrix[idx]
			                   + std::numeric_limits<float>::epsilon())
			            >= beta);
		}
	}

	std::complex<float> scale = {1.0f / ( float )nfft, 0.0};

	// apply masks to recover separated fft
	// recycle curr_fft instead of allocating a new vector
	std::size_t inverse_i = nfft - 1;
	for (std::size_t i = 0; i < nwin; ++i) {
		auto mask_idx = (stft_width - 1) * nwin + i;
		curr_fft[i] *= percussive_matrix[mask_idx] * scale;

		// P = P + flipud(conj(P))
		curr_fft[inverse_i - i] = std::conj(curr_fft[i]);
	}

	// apply ifft to get resultant percussive audio in-place
	// take the real part into the out arrays
	ffts_execute(fft_backward, curr_fft.data(), curr_fft.data());

	std::transform(
	    curr_fft.begin(), curr_fft.begin() + nwin, percussive_out_raw.begin(),
	    [](std::complex<float> x) -> float { return std::real(x); });

	// weighted overlap add with last iteration's samples - only half of the
	// real fft matters cola divide factor is for COLA compliance see
	// https://github.com/sevagh/Real-Time-HPSS for background
	for (std::size_t i = 0; i < nwin; ++i) {
		percussive_out[i] += percussive_out_raw[i] * COLA_factor;
	}

	// after weighted overlap add, the data we're ready to return
	// is the first 'hop' elements of harmonic_out and percussive_out
	// uses std::span (see hpss.h)
}
