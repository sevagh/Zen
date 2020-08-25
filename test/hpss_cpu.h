#ifndef HPSS_H
#define HPSS_H

#include "../src/window.h"
#include <complex>
#include <cstddef>
#include <fftw3.h>
#include <vector>

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 */
namespace rhythm_toolkit_cpu {
namespace hpss {
	class HPSS {
	public:
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		std::size_t stft_width;
		rhythm_toolkit_private::window::Window win;
		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		std::vector<std::complex<float>> sliding_stft;
		std::vector<std::complex<float>> curr_fft;

		std::vector<float> s_half_mag;
		std::vector<float> harmonic_matrix;
		std::vector<float> percussive_matrix;

		std::vector<float> input;

		std::vector<float> percussive_out_raw;
		std::vector<float> percussive_out;

		fftw_plan fft_forward;
		fftw_plan fft_backward;

		HPSS(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(std::size_t(ceilf(( float )l_harm / 2)))
		    , win(rhythm_toolkit_private::window::Window(
		          rhythm_toolkit_private::window::WindowType::SqrtVonHann,
		          nwin))
		    , COLA_factor(0.0f)
		    , sliding_stft(std::vector<std::complex<float>>(
		          stft_width * nfft,
		          std::complex<float>(0.0F, 0.0F)))
		    , curr_fft(std::vector<std::complex<float>>(nfft, 0.0F))
		    , s_half_mag(std::vector<float>(stft_width * nwin, 0.0F))
		    , harmonic_matrix(std::vector<float>(stft_width * nwin, 0.0F))
		    , percussive_matrix(std::vector<float>(stft_width * nwin, 0.0F))
		    , input(std::vector<float>(nwin, 0.0F))
		    , percussive_out_raw(std::vector<float>(nwin, 0.0F))
		    , percussive_out(std::vector<float>(nwin, 0.0F))
		{
			l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd

			fft_forward = fftw_plan_dft_1d(
			    nfft, reinterpret_cast<fftw_complex*>(curr_fft.data()),
			    reinterpret_cast<fftw_complex*>(curr_fft.data()), FFTW_FORWARD,
			    FFTW_MEASURE);
			fft_backward = fftw_plan_dft_1d(
			    nfft, reinterpret_cast<fftw_complex*>(curr_fft.data()),
			    reinterpret_cast<fftw_complex*>(curr_fft.data()),
			    FFTW_BACKWARD, FFTW_MEASURE);

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;
		};

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 512, 2.0){};

		~HPSS()
		{
			fftw_destroy_plan(fft_forward);
			fftw_destroy_plan(fft_backward);
		}

		void process_next_hop(std::vector<float>& current_hop);

		std::vector<float> peek_separated_percussive()
		{
			return std::vector<float>(
			    percussive_out.begin(), percussive_out.begin() + hop);
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_cpu

#endif /* HPSS_H */