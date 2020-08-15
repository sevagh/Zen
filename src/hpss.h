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
		thrust::device_vector<float> percussive_out;

		float COLA_factor; // see
		                   // https://www.mathworks.com/help/signal/ref/iscola.html

		// cufft specifics
		cuFloatComplex* fft_ptr;

		cufftHandle plan_forward;
		cufftHandle plan_backward;

		// npp specifics for median filtering
		NppStatus npp_status;
		cudaError cuda_status;

		NppiSize shared_roi;

		NppiSize harmonic_mask;
		NppiPoint harmonic_anchor;
		Npp8u* harmonic_buffer;
		Npp32u harmonic_buffer_size;

		NppiSize percussive_mask;
		NppiPoint percussive_anchor;
		Npp8u* percussive_buffer;
		Npp32u percussive_buffer_size;

		int nstep;

		HPSS(float fs, std::size_t hop, float beta)
		    : fs(fs)
		    , hop(hop)
		    , nwin(2*hop)
		    , nfft(4*hop)
		    , beta(beta)
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(std::size_t(ceilf(( float )l_harm / 2)))
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
		    , percussive_out(thrust::device_vector<float>(nwin, 0.0F))
		    , COLA_factor(0.0f)
		    , fft_ptr(
		          ( cuFloatComplex* )thrust::raw_pointer_cast(curr_fft.data()))
		    , shared_roi(NppiSize{stft_width, nfft})
		    , harmonic_mask(NppiSize{l_harm, 0})
		    , harmonic_anchor(NppiPoint{l_harm / 2 + 1, 0})
		    , percussive_mask(NppiSize{0, l_perc})
		    , percussive_anchor(NppiPoint{0, l_perc / 2 + 1})
		    , nstep(stft_width*sizeof(Npp32f))
		{
			l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd

			cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, 1);
			cufftPlan1d(&plan_backward, nfft, CUFFT_C2C, 1);

			// COLA = nfft/sum(win.*win)
			for (std::size_t i = 0; i < nwin; ++i) {
				COLA_factor += win.window[i] * win.window[i];
			}
			COLA_factor = nfft / COLA_factor;

			// set up median filtering buffers etc.
			npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
			    shared_roi, harmonic_mask, &harmonic_buffer_size);
			if (npp_status != NPP_NO_ERROR) {
				std::cerr << "NPP error " << npp_status << std::endl;
				std::exit(1);
			}

			npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
			    shared_roi, percussive_mask, &percussive_buffer_size);
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
		};

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 512, 2.0){};

		~HPSS()
		{
			cudaFree(harmonic_buffer);
			cudaFree(percussive_buffer);
		}

		void process_next_hop(std::vector<float>& current_hop);

		std::vector<float> peek_separated_percussive()
		{
			// this does cudamemcpy under the hood - use only for debugging!
			std::vector<float> ret(hop);

			// only the first hop samples of percussive_out are ready to be
			// consumed the rest remains to be overlap-added next iteration
			thrust::copy(percussive_out.begin(), percussive_out.begin() + hop,
			             ret.begin());
			return ret;
		}
	};
}; // namespace hpss
}; // namespace rhythm_toolkit_private

#endif /* HPSS_PRIVATE_H */
