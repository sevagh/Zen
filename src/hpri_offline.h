#ifndef HPRI_OFFLINE_PRIVATE_H
#define HPRI_OFFLINE_PRIVATE_H

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
	class HPROfflineGPU {
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
		thrust::device_vector<thrust::complex<float>> curr_fft;

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

		// cufft specifics
		cuFloatComplex* fft_ptr;

		cufftHandle plan_forward;
		cufftHandle plan_backward;

		median_filter::MedianFilterGPU time;
		median_filter::MedianFilterGPU frequency;

		HPROfflineGPU(float fs, std::size_t hop, float beta)
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
		    , curr_fft(thrust::device_vector<thrust::complex<float>>(nfft, 0.0F))
		    , s_mag(thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_mask(thrust::device_vector<float>(lag * nfft, 0.0F))
		    , harmonic_mask(thrust::device_vector<float>(lag * nfft, 0.0F))
		    , residual_mask(thrust::device_vector<float>(lag * nfft, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , residual_out(thrust::device_vector<float>(nwin, 0.0F))
		    , harmonic_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , fft_ptr(
		          ( cuFloatComplex* )thrust::raw_pointer_cast(curr_fft.data()))
		    , time(stft_width,
		           nfft,
		           l_harm,
		           median_filter::MedianFilterDirection::TimeAnticausal)
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

		void process_next_hop(thrust::device_ptr<float> in_hop,
		                      bool only_percussive = false);
	};

	class HPROfflineCPU {
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
		std::vector<thrust::complex<float>> curr_fft;

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

		// ipp fft specifics
		int fft_order;

		Ipp32fc* fft_ptr;
		IppsFFTSpec_C_32fc* fft_spec;

		Ipp8u* p_mem_spec;
		Ipp8u* p_mem_init;
		Ipp8u* p_mem_buffer;

		int size_spec;
		int size_init;
		int size_buffer;

		median_filter::MedianFilterCPU time;
		median_filter::MedianFilterCPU frequency;

		HPROfflineCPU(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , fft_order(( int )log2(nfft))
		    , beta(beta)
		    , l_harm(roundf(0.2 / (( float )(nfft - hop) / fs)))
		    , lag(l_harm)
		    , l_perc(roundf(500 / (fs / ( float )nfft)))
		    , stft_width(2 * l_harm)
		    , input(std::vector<float>(nwin, 0.0F))
		    , win(window::WindowCPU(window::WindowType::SqrtVonHann, nwin))
		    , sliding_stft(std::vector<thrust::complex<float>>(
		          stft_width * nfft,
		          thrust::complex<float>{0.0F, 0.0F}))
		    , curr_fft(std::vector<thrust::complex<float>>(nfft, 0.0F))
		    , s_mag(std::vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(std::vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(std::vector<float>(stft_width * nfft, 0.0F))
		    , percussive_mask(std::vector<float>(lag * nfft, 0.0F))
		    , harmonic_mask(std::vector<float>(lag * nfft, 0.0F))
		    , residual_mask(std::vector<float>(lag * nfft, 0.0F))
		    , percussive_out(std::vector<float>(nwin, 0.0F))
		    , residual_out(std::vector<float>(nwin, 0.0F))
		    , harmonic_out(std::vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , fft_ptr(( Ipp32fc* )curr_fft.data())
		    , time(stft_width,
		           nfft,
		           l_harm,
		           median_filter::MedianFilterDirection::TimeAnticausal)
		    , frequency(stft_width,
		                nfft,
		                l_perc,
		                median_filter::MedianFilterDirection::Frequency)
		    , p_mem_spec(nullptr)
		    , p_mem_init(nullptr)
		    , p_mem_buffer(nullptr)
		    , size_spec(0)
		    , size_init(0)
		    , size_buffer(0)
		{
			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			IppStatus ipp_status = ippsFFTGetSize_C_32fc(
			    fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &size_spec,
			    &size_init, &size_buffer);
			if (ipp_status != ippStsNoErr) {
				std::cerr << "ippFFTGetSize error: " << ipp_status << ", "
				          << ippGetStatusString(ipp_status) << std::endl;
				std::exit(-1);
			}

			if (size_init > 0)
				p_mem_init = ( Ipp8u* )ippMalloc(size_init);
			if (size_buffer > 0)
				p_mem_buffer = ( Ipp8u* )ippMalloc(size_buffer);
			if (size_spec > 0)
				p_mem_spec = ( Ipp8u* )ippMalloc(size_spec);

			ipp_status = ippsFFTInit_C_32fc(
			    &fft_spec, fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
			    p_mem_spec, p_mem_init);
			if (ipp_status != ippStsNoErr) {
				std::cerr << "ippFFTInit error: " << ipp_status << ", "
				          << ippGetStatusString(ipp_status) << std::endl;
				std::exit(-1);
			}

			if (size_init > 0)
				ippFree(p_mem_init);
		};

		~HPROfflineCPU()
		{
			if (size_buffer > 0)
				ippFree(p_mem_buffer);
			if (size_spec > 0)
				ippFree(p_mem_spec);
		}

		void process_next_hop(float* in_hop, bool only_percussive = false);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* HPRI_OFFLINE_PRIVATE_H */
