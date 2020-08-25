#ifndef HPSS_PRIVATE_H
#define HPSS_PRIVATE_H

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

#include "rhythm_toolkit/io.h"

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 */
namespace rhythm_toolkit_private {
namespace hpss {
	void cuda_hpss(thrust::host_vector<float>& x,
	               thrust::host_vector<float>& p);

	class HPSS {
	public:
		rhythm_toolkit::io::IO& io;
		float fs;
		std::size_t hop;
		std::size_t nwin;
		std::size_t nfft;
		float beta;
		int l_harm;
		int l_perc;
		std::size_t stft_width;

		thrust::device_vector<float> input;
		window::Window win;

		thrust::device_vector<thrust::complex<float>> sliding_stft;
		thrust::device_vector<thrust::complex<float>> curr_fft;

		thrust::device_vector<float> s_mag;
		thrust::device_vector<float> harmonic_matrix;
		thrust::device_vector<float> percussive_matrix;
		thrust::device_vector<float> harmonic_mask;
		thrust::device_vector<float> percussive_mask;
		thrust::device_vector<float> percussive_out;
		thrust::device_vector<float> harmonic_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		// cufft specifics
		cuFloatComplex* fft_ptr;

		cufftHandle plan_forward;
		cufftHandle plan_backward;

		// npp specifics for median filtering
		NppStatus npp_status;
		cudaError cuda_status;

		NppiSize medfilt_roi;

		int harmonic_roi_offset;
		NppiSize harmonic_filter_mask;
		NppiPoint harmonic_anchor;
		Npp8u* harmonic_buffer;
		Npp32u harmonic_buffer_size;

		int percussive_roi_offset;
		NppiSize percussive_filter_mask;
		NppiPoint percussive_anchor;
		Npp8u* percussive_buffer;
		Npp32u percussive_buffer_size;

		int nstep;

		HPSS(float fs, std::size_t hop, float beta, rhythm_toolkit::io::IO& io)
		    : io(io)
		    , fs(fs)
		    , hop(hop)
		    , nwin(2 * hop)
		    , nfft(4 * hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(l_harm)
		    , input(thrust::device_vector<float>(nwin, 0.0F))
		    , win(window::Window(window::WindowType::SqrtVonHann, nwin))
		    , sliding_stft(thrust::device_vector<thrust::complex<float>>(
		          stft_width * nfft,
		          thrust::complex<float>{0.0F, 0.0F}))
		    , curr_fft(thrust::device_vector<thrust::complex<float>>(nfft, 0.0F))
		    , s_mag(thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , percussive_matrix(
		          thrust::device_vector<float>(stft_width * nfft, 0.0F))
		    , harmonic_mask(thrust::device_vector<float>(nfft, 0.0F))
		    , percussive_mask(thrust::device_vector<float>(nfft, 0.0F))
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , harmonic_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , fft_ptr(
		          ( cuFloatComplex* )thrust::raw_pointer_cast(curr_fft.data()))
		    , nstep(nfft * sizeof(Npp32f))
		{
			l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			// set up median filtering buffers etc.
			medfilt_roi = NppiSize{nfft, stft_width};

			harmonic_roi_offset = ( int )floorf(( float )l_harm / 2);
			harmonic_filter_mask = NppiSize{1, l_harm};
			harmonic_anchor = NppiPoint{0, harmonic_roi_offset};

			percussive_roi_offset = ( int )floorf(( float )l_perc / 2);
			percussive_filter_mask = NppiSize{l_perc, 1};
			percussive_anchor = NppiPoint{percussive_roi_offset, 0};

			npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
			    medfilt_roi, harmonic_filter_mask, &harmonic_buffer_size);
			if (npp_status != NPP_NO_ERROR) {
				std::cerr << "NPP error " << npp_status << std::endl;
				std::exit(1);
			}

			npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
			    medfilt_roi, percussive_filter_mask, &percussive_buffer_size);
			if (npp_status != NPP_NO_ERROR) {
				std::cerr << "NPP error " << npp_status << std::endl;
				std::exit(1);
			}

			cuda_status
			    = cudaMalloc(( void** )&harmonic_buffer, harmonic_buffer_size);
			if (cuda_status != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_status);
				std::exit(1);
			}

			cuda_status = cudaMalloc(
			    ( void** )&percussive_buffer, percussive_buffer_size);
			if (cuda_status != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_status);
				std::exit(1);
			}

			cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, 1);
			cufftPlan1d(&plan_backward, nfft, CUFFT_C2C, 1);
		};

		// sensible defaults
		HPSS(float fs, rhythm_toolkit::io::IO& io)
		    : HPSS(fs, 512, 2.0, io){};

		~HPSS()
		{
			cudaFree(harmonic_buffer);
			cudaFree(percussive_buffer);
		}

		// copies data from the io object
		void process_next_hop();

		// populates the io object
		void peek_separated_percussive()
		{
			// only the first hop samples of percussive_out are ready to be
			// consumed the rest remains to be overlap-added next iteration
			thrust::copy(percussive_out.begin(), percussive_out.begin() + hop,
			             io.device_out);
		}

		void reset()
		{
			thrust::fill(input.begin(), input.end(), 0.0F);
			thrust::fill(sliding_stft.begin(), sliding_stft.end(),
			             thrust::complex<float>{0.0F, 0.0F});
			thrust::fill(curr_fft.begin(), curr_fft.end(),
			             thrust::complex<float>{0.0F, 0.0F});
			thrust::fill(s_mag.begin(), s_mag.end(), 0.0F);
			thrust::fill(harmonic_matrix.begin(), harmonic_matrix.end(), 0.0F);
			thrust::fill(
			    percussive_matrix.begin(), percussive_matrix.end(), 0.0F);
			thrust::fill(harmonic_mask.begin(), harmonic_mask.end(), 0.0F);
			thrust::fill(percussive_mask.begin(), percussive_mask.end(), 0.0F);
			thrust::fill(percussive_out.begin(), percussive_out.end(), 0.0F);
			thrust::fill(harmonic_out.begin(), harmonic_out.end(), 0.0F);
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* HPSS_PRIVATE_H */
