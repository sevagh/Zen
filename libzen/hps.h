#ifndef ZG_HPS_INTERNAL_H
#define ZG_HPS_INTERNAL_H

#include <box.h>
#include <complex>
#include <cstddef>
#include <fftw.h>
#include <mfilt.h>
#include <vector>
#include <win.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <core.h>
#include <libzen/hps.h>

namespace zen {
namespace internal {
	namespace hps {
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

		struct reciprocal_functor {
			const float factor;
			reciprocal_functor(float _factor)
			    : factor(_factor)
			{
			}

			__host__ __device__ float operator()(const float& x) const
			{
				return (1.0f / x) * factor;
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

			__host__ __device__ float
			operator()(const thrust::complex<float>& x, const float& y) const
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

		struct complex_abs_squared_functor {
			template <typename ValueType>
			__host__ __device__ ValueType
			operator()(const thrust::complex<ValueType>& z)
			{
				return powf(thrust::abs(z), 2.0f);
			}
		};

		struct hard_mask_functor {
			const float beta;

			hard_mask_functor(float _beta)
			    : beta(_beta)
			{
			}

			__host__ __device__ float operator()(const float& x,
			                                     const float& y) const
			{
				return float((x / (y + Eps)) >= beta);
			}
		};

		// soft/weiner mask
		struct soft_mask_functor {
			const int power;
			soft_mask_functor(int _power)
			    : power(_power)
			{
			}

			__host__ __device__ float operator()(const float& x,
			                                     const float& y) const
			{
				return float(powf(x, power)
				             / (powf(x, power) + powf(y, power) + Eps));
			}
		};

		// soft/weiner mask
		struct sse_mask_functor {
			sse_mask_functor() {}

			__host__ __device__ float operator()(const float& x,
			                                     const float& y) const
			{
				return float(x * x / (x * x + y * y + Eps));
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

		template <zen::Backend B>
		class HPR {

			typedef typename zen::internal::core::TypeTraits<B>::InputPointer
			    InputPointer;
			typedef
			    typename zen::internal::core::TypeTraits<B>::RealVector RealVector;
			typedef typename zen::internal::core::TypeTraits<B>::ComplexVector
			    ComplexVector;
			typedef typename zen::internal::core::TypeTraits<B>::MedianFilter
			    MedianFilter;
			typedef
			    typename zen::internal::core::TypeTraits<B>::BoxFilter BoxFilter;
			typedef typename zen::internal::core::TypeTraits<B>::FFTC2CWrapper
			    FFTC2CWrapper;
			typedef typename zen::internal::core::TypeTraits<B>::Window Window;

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

			RealVector input;
			Window window;

			ComplexVector sliding_stft;

			RealVector s_mag;
			RealVector reciprocal;
			RealVector harmonic_matrix;
			RealVector percussive_matrix;

			RealVector percussive_mask;
			RealVector harmonic_mask;
			RealVector residual_mask;
			RealVector percussive_out;
			RealVector harmonic_out;
			RealVector residual_out;

			float COLA_factor; // see
			    // https://www.mathworks.com/help/signal/ref/iscola.html

			MedianFilter time;
			MedianFilter frequency;

			BoxFilter time_sse;
			BoxFilter frequency_sse;

			FFTC2CWrapper fft;

			bool output_percussive;
			bool output_harmonic;
			bool output_residual;
			bool use_sse;
			bool soft_mask;

			HPR(float fs,
			    std::size_t hop,
			    float beta,
			    unsigned int output_flags,
			    mfilt::MedianFilterDirection causality,
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
			    , input(nwin, 0.0F)
			    , window(win::WindowType::SqrtVonHann, nwin)
			    , sliding_stft(stft_width * nfft,
			                   thrust::complex<float>{0.0F, 0.0F})
			    , s_mag(stft_width * nfft, 0.0F)
			    , reciprocal(stft_width * nfft, 0.0F)
			    , percussive_matrix(stft_width * nfft, 0.0F)
			    , harmonic_matrix(stft_width * nfft, 0.0F)
			    , percussive_mask(stft_width * nfft, 0.0F)
			    , harmonic_mask(stft_width * nfft, 0.0F)
			    , residual_mask(stft_width * nfft, 0.0F)
			    , percussive_out(nwin, 0.0F)
			    , residual_out(nwin, 0.0F)
			    , harmonic_out(nwin, 0.0F)
			    , COLA_factor(0.0f)
			    , time(stft_width, nfft, l_harm, causality, copy_bord)
			    , frequency(stft_width,
			                nfft,
			                l_perc,
			                mfilt::MedianFilterDirection::Frequency,
			                copy_bord)
			    , time_sse(stft_width, nfft, l_harm, causality)
			    , frequency_sse(stft_width,
			                    nfft,
			                    l_perc,
			                    mfilt::MedianFilterDirection::Frequency)
			    , fft(nfft)
			    , output_harmonic(false)
			    , output_percussive(false)
			    , output_residual(false)
			    , use_sse(false)
			    , soft_mask(false)
			{
				// causal = realtime
				if (causality == mfilt::MedianFilterDirection::TimeCausal) {
					// no lagging frames, output = latest frame
					lag = 1;
				}

				// COLA = nfft/sum(win.*win)
				for (std::size_t i = 0; i < nwin; ++i) {
					COLA_factor += window.window[i] * window.window[i];
				}
				COLA_factor = nfft / COLA_factor;

				if (output_flags & zen::hps::OUTPUT_HARMONIC) {
					output_harmonic = true;
				}
				if (output_flags & zen::hps::OUTPUT_PERCUSSIVE) {
					output_percussive = true;
				}
				if (output_flags & zen::hps::OUTPUT_RESIDUAL) {
					output_residual = true;
				}
			}

			void use_sse_filter() { use_sse = true; }

			void use_soft_mask() { soft_mask = true; }

			void process_next_hop(InputPointer in_hop);

			void apply_median_filter();
			void apply_sse_filter();

			void reset_buffers()
			{
				thrust::fill(input.begin(), input.end(), 0.0F);

				thrust::fill(
				    percussive_out.begin(), percussive_out.end(), 0.0F);
				thrust::fill(harmonic_out.begin(), harmonic_out.end(), 0.0F);
				thrust::fill(residual_out.begin(), residual_out.end(), 0.0F);

				thrust::fill(fft.fft_vec.begin(), fft.fft_vec.end(),
				             thrust::complex<float>{0.0F, 0.0F});
				thrust::fill(sliding_stft.begin(), sliding_stft.end(),
				             thrust::complex<float>{0.0F, 0.0F});

				thrust::fill(s_mag.begin(), s_mag.end(), 0.0F);
				thrust::fill(reciprocal.begin(), reciprocal.end(), 0.0F);
				thrust::fill(
				    harmonic_matrix.begin(), harmonic_matrix.end(), 0.0F);
				thrust::fill(
				    percussive_matrix.begin(), percussive_matrix.end(), 0.0F);

				thrust::fill(harmonic_mask.begin(), harmonic_mask.end(), 0.0F);
				thrust::fill(
				    percussive_mask.begin(), percussive_mask.end(), 0.0F);
				thrust::fill(residual_mask.begin(), residual_mask.end(), 0.0F);
			}
		};
	}; // namespace hps
};     // namespace internal
};     // namespace zen

#endif /* ZG_HPS_INTERNAL_H */
