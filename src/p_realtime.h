#ifndef P_REALTIME_PRIVATE_H
#define P_REALTIME_PRIVATE_H

#include "medianfilter.h"
#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "npp.h"
#include "nppdefs.h"
#include "nppi.h"

namespace rhythm_toolkit_private {
namespace hpss {
	class PRealtimeGPU {
	public:
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		std::size_t stft_width;

		thrust::device_vector<float> input;
		window::WindowGPU win;

		thrust::device_vector<thrust::complex<float>> sliding_stft;
		thrust::device_vector<thrust::complex<float>> curr_fft;

		thrust::device_vector<float> s_mag;
		thrust::device_vector<float> harmonic_matrix;
		thrust::device_vector<float> percussive_matrix;
		thrust::device_vector<float> percussive_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		// cufft specifics
		cuFloatComplex* fft_ptr;

		cufftHandle plan_forward;
		cufftHandle plan_backward;

		median_filter::MedianFilterGPU time;
		median_filter::MedianFilterGPU frequency;

		PRealtimeGPU(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(l_harm)
		    , input(thrust::device_vector<float>(nwin, 0.0F))
		    , win(window::WindowGPU(window::WindowType::SqrtVonHann, nwin))
		    , sliding_stft(thrust::device_vector<thrust::complex<float>>(
		          stft_width * nfft,
		          thrust::complex<float>{0.0F, 0.0F}))
		    , curr_fft(thrust::device_vector<thrust::complex<float>>(nfft, 0.0F))
		    , s_mag(thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , fft_ptr(
		          ( cuFloatComplex* )thrust::raw_pointer_cast(curr_fft.data()))
		    , time(stft_width,
		           nfft,
		           l_harm / 2,
		           median_filter::MedianFilterDirection::TimeCausal)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency)
		{
			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, 1);
			cufftPlan1d(&plan_backward, nfft, CUFFT_C2C, 1);
		};

		void process_next_hop(thrust::device_ptr<float> in_hop);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* P_REALTIME_PRIVATE_H */
