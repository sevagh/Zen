#ifndef ZG_FFT_INTERNAL_H
#define ZG_FFT_INTERNAL_H

#include <complex>
#include <cstddef>
#include <vector>

#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>

namespace zg {
namespace internal {
	namespace fftw {
		class FFTWrapperGPU {
		public:
			std::size_t nfft;

			thrust::device_vector<thrust::complex<float>> fft_vec;

			FFTWrapperGPU(std::size_t nfft)
			    : nfft(nfft)
			    , fft_vec(nfft)
			    , fft_ptr(( cuFloatComplex* )thrust::raw_pointer_cast(
			          fft_vec.data()))
			{
				cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, 1);
				cufftPlan1d(&plan_backward, nfft, CUFFT_C2C, 1);
			}

			void forward()
			{
				cufftExecC2C(plan_forward, fft_ptr, fft_ptr, CUFFT_FORWARD);
			}

			void backward()
			{
				cufftExecC2C(plan_backward, fft_ptr, fft_ptr, CUFFT_INVERSE);
			}

		private:
			cuFloatComplex* fft_ptr;

			cufftHandle plan_forward;
			cufftHandle plan_backward;
		};

		class FFTWrapperCPU {
		public:
			std::size_t nfft;

			std::vector<thrust::complex<float>> fft_vec;

			FFTWrapperCPU(std::size_t nfft)
			    : nfft(nfft)
			    , fft_order(( int )log2(nfft))
			    , fft_vec(nfft)
			    , fft_ptr(( Ipp32fc* )fft_vec.data())
			    , p_mem_spec(nullptr)
			    , p_mem_init(nullptr)
			    , p_mem_buffer(nullptr)
			    , size_spec(0)
			    , size_init(0)
			    , size_buffer(0)
			{
				IppStatus ipp_status = ippsFFTGetSize_C_32fc(
				    fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
				    &size_spec, &size_init, &size_buffer);
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
			}

			~FFTWrapperCPU()
			{
				if (size_buffer > 0)
					ippFree(p_mem_buffer);
				if (size_spec > 0)
					ippFree(p_mem_spec);
			}

			void forward()
			{
				ippsFFTFwd_CToC_32fc_I(fft_ptr, fft_spec, p_mem_buffer);
			}

			void backward()
			{
				ippsFFTInv_CToC_32fc_I(fft_ptr, fft_spec, p_mem_buffer);
			}

		private:
			int fft_order;

			Ipp32fc* fft_ptr;
			IppsFFTSpec_C_32fc* fft_spec;

			Ipp8u* p_mem_spec;
			Ipp8u* p_mem_init;
			Ipp8u* p_mem_buffer;

			int size_spec;
			int size_init;
			int size_buffer;
		};

	}; // namespace fftw
};     // namespace internal
};     // namespace zg

#endif /* ZG_FFT_INTERNAL_H */
