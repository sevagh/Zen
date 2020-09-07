#ifndef HPR_PRIVATE_H
#define HPR_PRIVATE_H

#include "fft_wrapper.h"
#include "medianfilter.h"
#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rhythm_toolkit_private {
namespace hpss {
	const unsigned int HPSS_HARMONIC = 1;
	const unsigned int HPSS_PERCUSSIVE = 1 << 1;
	const unsigned int HPSS_RESIDUAL = 1 << 2;

	class HPRGPU {
	public:
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		int lag; // lag specifies how far behind the output frame is compared
		         // to the tip in the anticausal case we're looking for l_harm
		         // frames backwards
		std::size_t stft_width;

		thrust::device_vector<float> input;
		window::WindowGPU win;

		thrust::device_vector<thrust::complex<float>> sliding_stft;

		thrust::device_vector<float> s_mag;
		thrust::device_vector<float> harmonic_matrix;
		thrust::device_vector<float> percussive_matrix;

		thrust::device_vector<float> percussive_mask;
		thrust::device_vector<float> harmonic_mask;
		thrust::device_vector<float> residual_mask;
		thrust::device_vector<float> percussive_out;
		thrust::device_vector<float> harmonic_out;
		thrust::device_vector<float> residual_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		median_filter::MedianFilterGPU time;
		median_filter::MedianFilterGPU frequency;

		fft_wrapper::FFTWrapperGPU fft;

		bool output_percussive;
		bool output_harmonic;
		bool output_residual;

		HPRGPU(float fs,
		       std::size_t hop,
		       float beta,
		       int output_flags,
		       median_filter::MedianFilterDirection causality,
		       bool copy_bord)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / (( float )(nfft - hop) / fs)))
		    , lag(l_harm)
		    , l_perc(roundf(500 / (fs / ( float )nfft)))
		    , stft_width(2 * l_harm)
		    , input(thrust::device_vector<float>(nwin, 0.0F))
		    , win(window::WindowGPU(window::WindowType::SqrtVonHann, nwin))
		    , sliding_stft(thrust::device_vector<thrust::complex<float>>(
		          stft_width * nfft,
		          thrust::complex<float>{0.0F, 0.0F}))
		    , s_mag(thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_mask(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_mask(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , residual_mask(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , residual_out(thrust::device_vector<float>(nwin, 0.0F))
		    , harmonic_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , time(stft_width, nfft, l_harm, causality, copy_bord)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency,
		                copy_bord)
		    , fft(fft_wrapper::FFTWrapperGPU(nfft))
		    , output_harmonic(false)
		    , output_percussive(false)
		    , output_residual(false)
		{
			// causal = realtime
			if (causality == median_filter::MedianFilterDirection::TimeCausal) {
				// no lagging frames, output = latest frame
				lag = 1;
			}

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			if (output_flags & HPSS_HARMONIC) {
				output_harmonic = true;
			}
			if (output_flags & HPSS_PERCUSSIVE) {
				output_percussive = true;
			}
			if (output_flags & HPSS_RESIDUAL) {
				output_residual = true;
			}
		};

		void process_next_hop(thrust::device_ptr<float> in_hop);

		void reset_buffers()
		{
			thrust::fill(input.begin(), input.end(), 0.0F);

			thrust::fill(percussive_out.begin(), percussive_out.end(), 0.0F);
			thrust::fill(harmonic_out.begin(), harmonic_out.end(), 0.0F);
			thrust::fill(residual_out.begin(), residual_out.end(), 0.0F);

			thrust::fill(fft.fft_vec.begin(), fft.fft_vec.end(),
			             thrust::complex<float>{0.0F, 0.0F});
			thrust::fill(sliding_stft.begin(), sliding_stft.end(),
			             thrust::complex<float>{0.0F, 0.0F});

			thrust::fill(s_mag.begin(), s_mag.end(), 0.0F);
			thrust::fill(harmonic_matrix.begin(), harmonic_matrix.end(), 0.0F);
			thrust::fill(
			    percussive_matrix.begin(), percussive_matrix.end(), 0.0F);

			thrust::fill(harmonic_mask.begin(), harmonic_mask.end(), 0.0F);
			thrust::fill(percussive_mask.begin(), percussive_mask.end(), 0.0F);
			thrust::fill(residual_mask.begin(), residual_mask.end(), 0.0F);
		}
	};

	class HPRCPU {
	public:
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		int lag; // lag specifies how far behind the output frame is compared
		         // to the tip in the anticausal case we're looking for l_harm
		         // frames backwards
		std::size_t stft_width;

		std::vector<float> input;
		window::WindowCPU win;

		std::vector<thrust::complex<float>> sliding_stft;

		std::vector<float> s_mag;
		std::vector<float> harmonic_matrix;
		std::vector<float> percussive_matrix;

		std::vector<float> percussive_mask;
		std::vector<float> harmonic_mask;
		std::vector<float> residual_mask;
		std::vector<float> percussive_out;
		std::vector<float> harmonic_out;
		std::vector<float> residual_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		median_filter::MedianFilterCPU time;
		median_filter::MedianFilterCPU frequency;

		fft_wrapper::FFTWrapperCPU fft;

		bool output_percussive;
		bool output_harmonic;
		bool output_residual;

		HPRCPU(float fs,
		       std::size_t hop,
		       float beta,
		       int output_flags,
		       median_filter::MedianFilterDirection causality)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / (( float )(nfft - hop) / fs)))
		    , lag(l_harm)
		    , l_perc(roundf(500 / (fs / ( float )nfft)))
		    , stft_width(2 * l_harm)
		    , input(nwin, 0.0F)
		    , win(window::WindowType::SqrtVonHann, nwin)
		    , sliding_stft(stft_width * nfft,
		                   thrust::complex<float>{0.0F, 0.0F})
		    , s_mag(stft_width * nfft, 0.0F)
		    , percussive_matrix(stft_width * nfft, 0.0F)
		    , harmonic_matrix(stft_width * nfft, 0.0F)
		    , percussive_mask(stft_width * nfft, 0.0F)
		    , harmonic_mask(stft_width * nfft, 0.0F)
		    , residual_mask(stft_width * nfft, 0.0F)
		    , percussive_out(nwin, 0.0F)
		    , residual_out(nwin, 0.0F)
		    , harmonic_out(nwin, 0.0F)
		    , COLA_factor(0.0f)
		    , time(stft_width,
		           nfft,
		           l_harm,
		           median_filter::MedianFilterDirection::TimeAnticausal)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency)
		    , fft(nfft)
		    , output_harmonic(false)
		    , output_percussive(false)
		    , output_residual(false)
		{
			// causal = realtime
			if (causality == median_filter::MedianFilterDirection::TimeCausal) {
				// no lagging frames, output = latest frame
				lag = 1;
			}

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			if (output_flags & HPSS_HARMONIC) {
				output_harmonic = true;
			}
			if (output_flags & HPSS_PERCUSSIVE) {
				output_percussive = true;
			}
			if (output_flags & HPSS_RESIDUAL) {
				output_residual = true;
			}
		};

		void process_next_hop(float* in_hop);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* HPR_PRIVATE_H */
