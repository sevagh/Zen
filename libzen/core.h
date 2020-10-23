#ifndef ZG_CORE_H
#define ZG_CORE_H

#include <fftw.h>
#include <hps/mfilt.h>
#include <hps/box.h>
#include <libzen/zen.h>
#include <stdexcept>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <vector>
#include <win.h>

namespace zen {
namespace internal {
	namespace core {
		template <zen::Backend T>
		struct TypeTraits {
		};

		template <>
		struct TypeTraits<zen::Backend::GPU> {
			typedef thrust::device_ptr<float> InputPointer;
			typedef thrust::device_vector<float> RealVector;
			typedef thrust::device_vector<thrust::complex<float>> ComplexVector;
			typedef zen::internal::fftw::FFTC2CWrapperGPU FFTC2CWrapper;
			typedef zen::internal::hps::mfilt::MedianFilterGPU MedianFilter;
			typedef zen::internal::hps::box::BoxFilterGPU BoxFilter;
			typedef zen::internal::win::WindowGPU Window;
		};

		template <>
		struct TypeTraits<zen::Backend::CPU> {
			typedef float* InputPointer;
			typedef std::vector<float> RealVector;
			typedef std::vector<thrust::complex<float>> ComplexVector;
			typedef zen::internal::fftw::FFTC2CWrapperCPU FFTC2CWrapper;
			typedef zen::internal::hps::mfilt::MedianFilterCPU MedianFilter;
			typedef zen::internal::hps::box::BoxFilterCPU BoxFilter;
			typedef zen::internal::win::WindowCPU Window;
		};
	}; // namespace core
};     // namespace internal
};     // namespace zen

#endif /* ZG_CORE_H */
