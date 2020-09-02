#ifndef HPSS_PRIVATE_H
#define HPSS_PRIVATE_H

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

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 *
 * 2 classes:
 *
 * "HPROffline" - Harmonic-Percussive-Residual Offline
 * 	this one uses past, present, and future frames for optimal separation of
 * the harmonic it's contained in the public class HPRIOffline, i.e.
 * Harmonic-Percussive-Residual-Iterative, the two-pass method
 *
 * "PRealtime" - Percussive Realtime
 * 	this one only uses present and past frames to respect causality for
 * real-time purposes as only the percussive separation can sound good under
 * such conditions, it omits the computation of the harmonic and residual
 *
 * 	iterative is also impossible here due to the performance constraints of
 * real-time
 */

namespace rhythm_toolkit_private {
namespace hpss {
	static constexpr float Eps = std::numeric_limits<float>::epsilon();

	// various thrust GPU functors, reused across HPRIOfflineGPU and PRealtimeGPU
	struct window_functor {
		window_functor() {}

		__host__ __device__ thrust::complex<float>
		operator()(const float& x, const float& y) const
		{
			return thrust::complex<float>{x * y, 0.0};
		}
	};

	struct residual_mask_functor {
		residual_mask_functor() {}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return 1 - (x + y);
		}
	};

	struct apply_mask_functor {
		apply_mask_functor() {}

		__host__ __device__ thrust::complex<float>
		operator()(const thrust::complex<float>& x, const float& y) const
		{
			return x * y;
		}
	};

	struct overlap_add_functor {
		const float cola_factor;
		overlap_add_functor(float _cola_factor)
		    : cola_factor(_cola_factor)
		{
		}

		__host__ __device__ float operator()(const thrust::complex<float>& x,
		                                     const float& y) const
		{
			return y + x.real() * cola_factor;
		}
	};

	struct complex_abs_functor {
		template <typename ValueType>
		__host__ __device__ ValueType
		operator()(const thrust::complex<ValueType>& z)
		{
			return thrust::abs(z);
		}
	};

	struct mask_functor {
		const float beta;

		mask_functor(float _beta)
		    : beta(_beta)
		{
		}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return float((x / (y + Eps)) >= beta);
		}
	};

	struct sum_vectors_functor {
		sum_vectors_functor() {}

		__host__ __device__ float operator()(const float& x,
		                                     const float& y) const
		{
			return x + y;
		}
	};

	struct spectral_subtraction_functor {
		spectral_subtraction_functor() {}

		__host__ __device__ thrust::complex<float>
		operator()(const thrust::complex<float>& x,
		           const thrust::complex<float>& y) const
		{
			float mag = thrust::abs(x);
			float mag_noise = thrust::abs(y);

			// clip negative values to 0
			mag = mag > mag_noise ? mag - mag_noise : 0;

			// apply original phase + corrected magnitude
			return thrust::polar(mag, thrust::arg(x));
			// return thrust::complex<float>{x.real() - y, x.imag() - y};
		}
	};

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

#endif /* HPSS_PRIVATE_H */
