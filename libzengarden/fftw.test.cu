#include <gtest/gtest.h>
#include <fftw.h>
#include <iostream>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

using namespace zg::internal::fftw;
using namespace zg;

class FFTWrapperGPUTest : public ::testing::Test {

public:
	FFTWrapperGPU fftg;

	FFTWrapperGPUTest(int nfft)
	    : fftg(nfft) {};

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class FFTWrapperGPUTestSmall : public FFTWrapperGPUTest {
protected:
	FFTWrapperGPUTestSmall()
	    : FFTWrapperGPUTest(1024)
	{
	}
};

TEST_F(FFTWrapperGPUTestSmall, C2CForward)
{
	thrust::fill(fftg.fft_vec.begin(), fftg.fft_vec.end(), thrust::complex<float>{13.37F, -34.5F});

	std::cout << fftg.fft_vec[0] << "\n";

	fftg.forward();

	std::cout << fftg.fft_vec[0] << "\n";
}
