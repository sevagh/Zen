#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <complex>
#include <cstddef>
#include <ffts/ffts.h>
#include <stdexcept>
#include <vector>

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * don't need to maintain a sliding STFT matrix, and simply do a 1D median
 * filter on the FFT before applying the IFFT.
 */
namespace rhythm_toolkit {
namespace hpss {
	class HPSSException : public std::runtime_error {
	public:
		HPSSException(std::string msg)
		    : std::runtime_error(msg){};
	};

	class HPSS {
	public:
		float fs;
		std::size_t nwin;
		std::size_t nfft;
		std::size_t hop;
		float beta;
		float eps;
		int l_harm;
		int l_perc;
		std::size_t stft_width;
		window::Window win;
		float COLA_factor; // see https://www.mathworks.com/help/signal/ref/iscola.html

		// use 1d vectors for 2d
		std::vector<std::complex<float>> sliding_stft; // 1D sliding STFT
		                                               // matrix on which to
		                                               // apply median
		                                               // filtering and IFFT

		std::vector<std::complex<float>> curr_fft;     // store the fft of the
		                                               // current frame
		std::vector<std::complex<float>> harmonic_fft;
		std::vector<std::complex<float>> percussive_fft;
		std::vector<float> s_half_mag;
		std::vector<float> harmonic_matrix;
		std::vector<float> percussive_matrix;

		std::vector<float> harmonic_mask;
		std::vector<float> percussive_mask;

		std::vector<float> input;
		std::vector<float> input_windowed;

		std::vector<float> harmonic_out_raw;
		std::vector<float> percussive_out_raw;

		std::vector<float> harmonic_out;
		std::vector<float> percussive_out;

		std::vector<float> harmonic_out_hop;
		std::vector<float> percussive_out_hop;

		ffts_plan_t* fft_forward;
		ffts_plan_t* fft_backward;

		HPSS(float fs,
		     std::size_t nwin,
		     std::size_t nfft,
		     std::size_t hop,
		     float beta)
		    : fs(fs)
		    , nwin(nwin)
		    , nfft(nfft)
		    , hop(hop)
		    , beta(beta)
		    , eps(std::numeric_limits<float>::epsilon())
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(std::size_t(ceilf(( float )l_harm / 2)))
		    , win(window::Window(window::WindowType::SqrtVonHann, nwin))
		    , COLA_factor(0.0f)
		    , sliding_stft(
		          std::vector<std::complex<float>>(stft_width * nfft))
		    , curr_fft(std::vector<std::complex<float>>(nfft))
		    , harmonic_fft(std::vector<std::complex<float>>(nfft))
		    , percussive_fft(std::vector<std::complex<float>>(nfft))
		    , s_half_mag(std::vector<float>(stft_width * nwin))
		    , harmonic_matrix(std::vector<float>(stft_width * nwin))
		    , percussive_matrix(std::vector<float>(stft_width * nwin))
		    , harmonic_mask(std::vector<float>(stft_width * nwin))
		    , percussive_mask(std::vector<float>(stft_width * nwin))
		    , input(std::vector<float>(nwin))
		    , input_windowed(std::vector<float>(nwin))
		    , harmonic_out_raw(std::vector<float>(nfft))
		    , percussive_out_raw(std::vector<float>(nfft))
		    , harmonic_out(std::vector<float>(nwin))
		    , percussive_out(std::vector<float>(nwin))
		    , harmonic_out_hop(std::vector<float>(hop))
		    , percussive_out_hop(std::vector<float>(hop))
		{
			l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd

			fft_forward = ffts_init_1d(nfft, FFTS_FORWARD);
			fft_backward = ffts_init_1d(nfft, FFTS_BACKWARD);

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;
		};

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 1024, 2048, 512, 2.0){};
		// todo - nfft = 2 * nwin = 4 * hop - only require user to specify one,
		// not all 3

		~HPSS()
		{
			ffts_free(fft_forward);
			ffts_free(fft_backward);
		}

		void process_next_hop(std::vector<float>& current_hop);

		std::vector<float>& peek_separated_percussive()
		{
			return percussive_out_hop;
		}

		std::vector<float>& peek_separated_harmonic()
		{
			return harmonic_out_hop;
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
