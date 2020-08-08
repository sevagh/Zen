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

	// apply von hann window to input
	for (std::size_t i = 0; i < nwin; ++i) {
		input[i] = input[i] * win.window[i];
	}

	// rotate stft matrix to append newest FFT to end
	std::rotate(
	    sliding_stft.begin(), sliding_stft.begin() + 1, sliding_stft.end());

	// do the forward fft and store it in curr_fft
	std::transform(input.begin(), input.begin() + nwin, curr_fft.begin(),
	               [](float x) -> std::complex<float> {
		               return std::complex<float>(x, 0.0);
	               });

	// store real input as complex in last column of stft
	std::copy(curr_fft.begin(), curr_fft.end(),
	          sliding_stft.begin() + (stft_width - 1) * nfft);

	// forward fft in place
	ffts_execute(fft_forward, sliding_stft.data() + (stft_width - 1) * nfft,
	             sliding_stft.data() + (stft_width - 1) * nfft);

	// TODO: pragma omp stuff here? parallelize these loops
	// calculate magnitude of the stft
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nfft; ++j) {
			auto idx = i * nfft + j;
			auto elem = sliding_stft[idx];
			// |complex| = sqrt(r*r + i*i)
			s_half_mag[idx] = sqrtf(std::real(elem) * std::real(elem)
			                        + std::imag(elem) * std::imag(elem));
		}
	}

	// apply median filter in horizontal and vertical directions
	median_filter_2d<float>(( int )stft_width, ( int )nfft, 1, ( int )l_harm,
	                        0, s_half_mag.data(), harmonic_matrix.data());
	median_filter_2d<float>(( int )stft_width, ( int )nfft, ( int )l_perc, 1,
	                        0, s_half_mag.data(), percussive_matrix.data());

	// calculate the binary masks
	for (std::size_t i = 0; i < stft_width; ++i) {
		for (std::size_t j = 0; j < nfft; ++j) {
			auto idx = i * nfft + j;
			harmonic_mask[idx] = int(
			    harmonic_matrix[idx] / (percussive_matrix[idx] + eps) > beta);
			percussive_mask[idx] = int(
			    percussive_matrix[idx] / (harmonic_matrix[idx] + eps) >= beta);
		}
	}

	// apply masks to get separated fft
	for (std::size_t i = 0; i < nfft; ++i) {
		harmonic_fft[i] = {
		    harmonic_mask[(stft_width - 1) * nfft + i] * std::real(curr_fft[i]),
		    harmonic_mask[(stft_width - 1) * nfft + i] * std::imag(curr_fft[i]),
		};
		percussive_fft[i] = {
		    percussive_mask[(stft_width - 1) * nfft + i]
		        * std::real(curr_fft[i]),
		    percussive_mask[(stft_width - 1) * nfft + i]
		        * std::imag(curr_fft[i]),
		};
	}

	// apply ifft to get resultant audio
	ffts_execute(fft_backward, harmonic_fft.data(), harmonic_out_im.data());
	ffts_execute(fft_backward, percussive_fft.data(), percussive_out_im.data());

	// modify it by the window for COLA compliance
	for (std::size_t i = 0; i < nfft; ++i) {
		harmonic_fft[i] = harmonic_fft[i] / cola_divide_factor;
		percussive_fft[i] = percussive_fft[i] / cola_divide_factor;
	}

	// store only the real parts
	// only half fft matters
	std::transform(harmonic_out_im.begin(), harmonic_out_im.begin() + nwin,
	               harmonic_out.begin(), [](std::complex<float> cplx) -> float {
		               return std::real(cplx);
	               });

	// only half fft matters
	std::transform(
	    percussive_out_im.begin(), percussive_out_im.begin() + nwin,
	    percussive_out.begin(),
	    [](std::complex<float> cplx) -> float { return std::real(cplx); });
}
