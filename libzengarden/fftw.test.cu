#include <fftw.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

#include <cmath>

using namespace zg::internal::fftw;

// allow minor differences in cufft and ipp results
static constexpr float AllowableFFTError = 0.0002;

static std::vector<thrust::complex<float>> generate_data_normalized(size_t size)
{
	// use realistic values of normalized input floats
	static std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
	static std::default_random_engine generator;

	std::vector<float> data_im(size);
	std::vector<float> data_re(size);

	std::generate(data_re.begin(), data_re.end(),
	              []() { return distribution(generator); });
	std::generate(data_im.begin(), data_im.end(),
	              []() { return distribution(generator); });

	std::vector<thrust::complex<float>> data(size);

	thrust::transform(
	    data_re.begin(), data_re.end(), data_im.begin(), data.begin(),
	    [](const float& x, const float& y) -> thrust::complex<float> {
		    return thrust::complex<float>{x, y};
	    });
	return data;
}

static std::vector<thrust::complex<float>> generate_data_huge(size_t size)
{
	// use realistic values of normalized input floats
	static std::uniform_real_distribution<float> distribution(
	    std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
	static std::default_random_engine generator;

	std::vector<float> data_im(size);
	std::vector<float> data_re(size);

	std::generate(data_re.begin(), data_re.end(),
	              []() { return distribution(generator); });
	std::generate(data_im.begin(), data_im.end(),
	              []() { return distribution(generator); });

	std::vector<thrust::complex<float>> data(size);

	thrust::transform(
	    data_re.begin(), data_re.end(), data_im.begin(), data.begin(),
	    [](const float& x, const float& y) -> thrust::complex<float> {
		    return thrust::complex<float>{x, y};
	    });
	return data;
}

class FFTC2CWrapperTest : public ::testing::Test {

public:
	std::size_t nfft;
	FFTC2CWrapperGPU fftg;
	FFTC2CWrapperCPU fftc;
	std::vector<thrust::complex<float>> testdata_small;
	std::vector<thrust::complex<float>> testdata_huge;

	FFTC2CWrapperTest(std::size_t nfft)
	    : nfft(nfft)
	    , fftg(nfft)
	    , fftc(nfft)
	    , testdata_small(generate_data_normalized(nfft))
	    , testdata_huge(generate_data_huge(nfft)){};

	void cmp()
	{
		for (std::size_t i = 0; i < nfft; ++i) {
			auto gpu_real = (( thrust::complex<float> )fftc.fft_vec[i]).real();
			auto cpu_real = (( thrust::complex<float> )fftg.fft_vec[i]).real();

			auto gpu_imag = (( thrust::complex<float> )fftc.fft_vec[i]).imag();
			auto cpu_imag = (( thrust::complex<float> )fftg.fft_vec[i]).imag();

			// if both are nan or inf, don't really care
			EXPECT_TRUE(std::isfinite(gpu_real) == std::isfinite(cpu_real));

			if (std::isfinite(gpu_real))
				EXPECT_NEAR(gpu_real, cpu_real, AllowableFFTError);

			if (std::isfinite(gpu_imag))
				EXPECT_NEAR(gpu_imag, cpu_imag, AllowableFFTError);
		}
	}

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class FFTC2CWrapperTestVerySmall : public FFTC2CWrapperTest {
protected:
	FFTC2CWrapperTestVerySmall()
	    : FFTC2CWrapperTest(64)
	{
	}
};

class FFTC2CWrapperTestSmall : public FFTC2CWrapperTest {
protected:
	FFTC2CWrapperTestSmall()
	    : FFTC2CWrapperTest(1024)
	{
	}
};

class FFTC2CWrapperTestLarge : public FFTC2CWrapperTest {
protected:
	FFTC2CWrapperTestLarge()
	    : FFTC2CWrapperTest(16384)
	{
	}
};

TEST_F(FFTC2CWrapperTestVerySmall, C2CForwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestSmall, C2CForwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestLarge, C2CForwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestVerySmall, C2CForwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestSmall, C2CForwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestLarge, C2CForwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.forward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.forward();

	cmp();
}

TEST_F(FFTC2CWrapperTestVerySmall, C2CBackwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}

TEST_F(FFTC2CWrapperTestSmall, C2CBackwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}

TEST_F(FFTC2CWrapperTestLarge, C2CBackwardCompareIPPcuFFTSmallRange)
{
	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_small.begin(), testdata_small.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}

TEST_F(FFTC2CWrapperTestVerySmall, C2CBackwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}

TEST_F(FFTC2CWrapperTestSmall, C2CBackwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}

TEST_F(FFTC2CWrapperTestLarge, C2CBackwardCompareIPPcuFFTHugeRange)
{
	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftg.fft_vec.begin());
	fftg.backward();

	thrust::copy(
	    testdata_huge.begin(), testdata_huge.end(), fftc.fft_vec.begin());
	fftc.backward();

	cmp();
}
