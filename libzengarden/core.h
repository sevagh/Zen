#ifndef ZG_CORE_H
#define ZG_CORE_H

#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <vector>
#include <fftw.h>
#include <win.h>
#include <hps/mfilt.h>

namespace zg {
namespace internal {
namespace core {

enum Backend {
	GPU,
	CPU
};

template<Backend T>
struct TypeTraits {}; 

struct TypeTraits<Backend::GPU> {
	typedef thrust::device_ptr<float> InputPointer;
	typedef thrust::device_vector<float> RealVector;
	typedef thrust::device_vector<thrust::complex<float>> ComplexVector;
	typedef zg::internal::fftw::FFTWrapperGPU FFTWrapper;
	typedef zg::internal::hps::mfilt::MedianFilterGPU MedianFilter;
	typedef zg::internal::win::WindowGPU Window;
};

struct TypeTraits<Backend::CPU> {
	typedef float* InputPointer;
	typedef std::vector<float> RealVector;
	typedef std::vector<thrust::complex<float>> ComplexVector;
	typedef zg::internal::fftw::FFTWrapperCPU FFTWrapper;
	typedef zg::internal::hps::mfilt::MedianFilterCPU MedianFilter;
	typedef zg::internal::win::WindowCPU Window;
};
}; // namespace core
}; // namespace internal
}; // namespace zg

#endif /* ZG_CORE_H */
