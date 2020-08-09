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

	/******************************
	 * SO FAR IDENTICAL TO MATLAB *
	 ******************************/

	// do the real forward fft, store result in curr_fft
	ffts_execute(fft_forward, input_windowed.data(), curr_fft.data());

	// rotate stft matrix to move the oldest column to the end
	std::rotate(sliding_stft.begin(), sliding_stft.begin() + (nwin + 1),
	            sliding_stft.end());

	// copy curr_fft into the last column of the stft
	std::copy(curr_fft.begin(), curr_fft.end(),
	          sliding_stft.begin() + (stft_width - 1) * (nwin + 1));

	// calculate magnitude of the stft
	// TODO: pragma omp stuff here? parallelize these loops
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin + 1; ++j) {
			auto idx = i * (nwin + 1) + j;
			auto elem = sliding_stft[idx];
			// |complex| = sqrt(r*r + i*i)
			s_half_mag[idx] = sqrtf(std::real(elem) * std::real(elem)
			                        + std::imag(elem) * std::imag(elem));
		}
	}

	// apply median filter in horizontal and vertical directions
	// only consider half the stft
	median_filter_2d<float>(( int )stft_width, ( int )(nwin + 1), 1,
	                        ( int )l_harm, 0, s_half_mag.data(),
	                        harmonic_matrix.data());
	median_filter_2d<float>(( int )stft_width, ( int )(nwin + 1), ( int )l_perc,
	                        1, 0, s_half_mag.data(), percussive_matrix.data());

	// calculate the binary masks
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nwin + 1; ++j) {
			auto idx = i * (nwin + 1) + j;
			// TODO: comparing complex and float here is questionable, right?
			harmonic_mask[idx]
			    = int(std::real(harmonic_matrix[idx])
			              / (std::real(percussive_matrix[idx]) + eps)
			          > beta);
			percussive_mask[idx]
			    = int(std::real(percussive_matrix[idx])
			              / (std::real(harmonic_matrix[idx]) + eps)
			          >= beta);
		}
	}

	// apply masks to get separated fft
	for (std::size_t i = 0; i < nwin + 1; ++i) {
		harmonic_fft[i] = {
		    harmonic_mask[(stft_width - 1) * (nwin + 1) + i]
		        * std::real(curr_fft[i]),
		    harmonic_mask[(stft_width - 1) * (nwin + 1) + i]
		        * std::imag(curr_fft[i]),
		};
		percussive_fft[i] = {
		    percussive_mask[(stft_width - 1) * (nwin + 1) + i]
		        * std::real(curr_fft[i]),
		    percussive_mask[(stft_width - 1) * (nwin + 1) + i]
		        * std::imag(curr_fft[i]),
		};
	}

	// apply ifft to get resultant audio
	ffts_execute(fft_backward, harmonic_fft.data(), harmonic_out.data());
	ffts_execute(fft_backward, percussive_fft.data(), percussive_out.data());

	// weighted overlap add - only half of the real fft matters
	// cola divide factor is for COLA compliance
	// see https://github.com/sevagh/Real-Time-HPSS for background
	for (std::size_t i = 0; i < nwin; ++i) {
		harmonic_out[i] /= cola_divide_factor;
		percussive_out[i] /= cola_divide_factor;

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
}
