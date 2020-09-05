#ifndef P_REALTIME_PRIVATE_H
#define P_REALTIME_PRIVATE_H

#include "fft_wrapper.h"
#include "medianfilter.h"
#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

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

		thrust::device_vector<float> s_mag;
		thrust::device_vector<float> harmonic_matrix;
		thrust::device_vector<float> percussive_matrix;
		thrust::device_vector<float> percussive_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		median_filter::MedianFilterGPU time;
		median_filter::MedianFilterGPU frequency;
		fft_wrapper::FFTWrapperGPU fft;

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
		    , s_mag(thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , time(stft_width,
		           nfft,
		           l_harm / 2,
		           median_filter::MedianFilterDirection::TimeCausal)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency)
		    , fft(fft_wrapper::FFTWrapperGPU(nfft))
		{
			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;
		};

		void process_next_hop(thrust::device_ptr<float> in_hop);
	};

	class PRealtimeCPU {
	public:
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		std::size_t stft_width;

		std::vector<float> input;
		window::WindowCPU win;

		std::vector<thrust::complex<float>> sliding_stft;

		std::vector<float> s_mag;
		std::vector<float> harmonic_matrix;
		std::vector<float> percussive_matrix;
		std::vector<float> percussive_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		median_filter::MedianFilterCPU time;
		median_filter::MedianFilterCPU frequency;
		fft_wrapper::FFTWrapperCPU fft;

		PRealtimeCPU(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(l_harm)
		    , input(std::vector<float>(nwin, 0.0F))
		    , win(window::WindowCPU(window::WindowType::SqrtVonHann, nwin))
		    , sliding_stft(std::vector<thrust::complex<float>>(
		          stft_width * nfft,
		          thrust::complex<float>{0.0F, 0.0F}))
		    , s_mag(std::vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(std::vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(std::vector<float>(stft_width * nfft, 0.0F))
		    , percussive_out(std::vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , time(stft_width,
		           nfft,
		           l_harm / 2,
		           median_filter::MedianFilterDirection::TimeCausal)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency)
		    , fft(fft_wrapper::FFTWrapperCPU(nfft))
		{
			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;
		};

		void process_next_hop(const float* in_hop);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* P_REALTIME_PRIVATE_H */
