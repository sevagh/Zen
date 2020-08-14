#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cufft.h>

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 */
namespace rhythm_toolkit {
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
		window::Window win;
		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		thrust::device_vector<float> input;

		cufftReal* in_ptr;
		cufftReal* out_ptr;
		cuFloatComplex* fft_ptr;

		thrust::device_vector<thrust::complex<float>> sliding_stft;
		thrust::device_vector<thrust::complex<float>> curr_fft;

		thrust::device_vector<float> s_half_mag;
		thrust::device_vector<float> harmonic_matrix;
		thrust::device_vector<float> percussive_matrix;

		thrust::device_vector<float> percussive_out;

		cufftHandle plan_forward;
		cufftHandle plan_backward;

		HPSS(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(hop + 1)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(std::size_t(ceilf(( float )l_harm / 2)))
		    , win(window::Window(window::WindowType::SqrtVonHann, nwin))
		    , COLA_factor(0.0f)
		    , sliding_stft(
		          thrust::device_vector<thrust::complex<float>>(stft_width * nfft,
		                                           thrust::complex<float>{0.0F, 0.0F}))
		    , curr_fft(thrust::device_vector<thrust::complex<float>>(nfft, 0.0F))
		    , s_half_mag(thrust::device_vector<float>(stft_width * nwin, 0.0F))
		    , harmonic_matrix(thrust::device_vector<float>(stft_width * nwin, 0.0F))
		    , percussive_matrix(thrust::device_vector<float>(stft_width * nwin, 0.0F))
		    , input(thrust::device_vector<float>(nwin, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , in_ptr((cufftReal*)thrust::raw_pointer_cast(input.data()))
		    , out_ptr((cufftReal*)thrust::raw_pointer_cast(percussive_out.data()))
		    , fft_ptr((cuFloatComplex*)thrust::raw_pointer_cast(curr_fft.data()))
		{
			l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd

			cufftPlan1d(&plan_forward, nwin, CUFFT_R2C, 1);
			cufftPlan1d(&plan_backward, nwin, CUFFT_C2R, 1);

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;
		};

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 512, 2.0){};

		void process_next_hop(std::vector<float>& current_hop);

		// recall that only the first hop samples are useful!
		thrust::host_vector<float> peek_separated_percussive()
		{
			return thrust::host_vector<float>(percussive_out);
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
