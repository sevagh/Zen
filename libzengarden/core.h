#ifndef ZG_CORE_H
#define ZG_CORE_H

#include <fftw.h>
#include <hps/mfilt.h>
#include <libzengarden/zg.h>
#include <stdexcept>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <vector>
#include <win.h>

namespace zg {
namespace internal {
	namespace core {

		template <zg::Backend T>
		struct TypeTraits {
		};

		template <>
		struct TypeTraits<zg::Backend::GPU> {
			typedef thrust::device_ptr<float> InputPointer;
			typedef thrust::device_vector<float> RealVector;
			typedef thrust::device_vector<thrust::complex<float>> ComplexVector;
			typedef zg::internal::fftw::FFTWrapperGPU FFTWrapper;
			typedef zg::internal::hps::mfilt::MedianFilterGPU MedianFilter;
			typedef zg::internal::win::WindowGPU Window;
		};

		template <>
		struct TypeTraits<zg::Backend::CPU> {
			typedef float* InputPointer;
			typedef std::vector<float> RealVector;
			typedef std::vector<thrust::complex<float>> ComplexVector;
			typedef zg::internal::fftw::FFTWrapperCPU FFTWrapper;
			typedef zg::internal::hps::mfilt::MedianFilterCPU MedianFilter;
			typedef zg::internal::win::WindowCPU Window;
		};
	}; // namespace core
};     // namespace internal
};     // namespace zg

#endif /* ZG_CORE_H */
