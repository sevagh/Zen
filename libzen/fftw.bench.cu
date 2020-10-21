#include <benchmark/benchmark.h>

#include <fftw.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

#include <cmath>

using namespace zen::internal::fftw;

static std::vector<thrust::complex<float>> generate_data(size_t size)
{
	// use realistic values of normalized input floats
	static std::uniform_real_distribution<float> distribution(
	    std::numeric_limits<float>::min() / 2,
	    std::numeric_limits<float>::max() / 2);
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

static void BM_FFT_FORWARD_GPUCUFFT_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	// copy data outside of benchmark loop
	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.forward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_FORWARD_GPUCUFFT_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		// copy data inside of benchmark loop
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.forward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_FORWARD_CPUIPP_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.forward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_FORWARD_CPUIPP_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.forward();
	}
}

static void BM_FFT_BACKWARD_GPUCUFFT_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	// copy data outside of benchmark loop
	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_BACKWARD_GPUCUFFT_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		// copy data inside of benchmark loop
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_BACKWARD_CPUIPP_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_BACKWARD_CPUIPP_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.backward();
	}
}

static void BM_FFT_ROUNDTRIP_GPUCUFFT_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	// copy data outside of benchmark loop
	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.forward();
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_ROUNDTRIP_GPUCUFFT_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperGPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		// copy data inside of benchmark loop
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.forward();
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_ROUNDTRIP_CPUIPP_MEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	for (auto _ : state) {
		thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());
		fft.forward();
		fft.backward();
	}

	state.SetComplexityN(state.range(0));
}

static void BM_FFT_ROUNDTRIP_CPUIPP_NOMEM(benchmark::State& state)
{
	auto dim = state.range(0);

	auto fft = FFTC2CWrapperCPU(dim);
	auto data = generate_data(dim);

	thrust::copy(data.begin(), data.end(), fft.fft_vec.begin());

	for (auto _ : state) {
		fft.forward();
		fft.backward();
	}
}

BENCHMARK(BM_FFT_FORWARD_GPUCUFFT_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_FORWARD_GPUCUFFT_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_FORWARD_CPUIPP_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_FORWARD_CPUIPP_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();

BENCHMARK(BM_FFT_BACKWARD_GPUCUFFT_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_BACKWARD_GPUCUFFT_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_BACKWARD_CPUIPP_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_BACKWARD_CPUIPP_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();

BENCHMARK(BM_FFT_ROUNDTRIP_GPUCUFFT_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_ROUNDTRIP_GPUCUFFT_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_ROUNDTRIP_CPUIPP_NOMEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();
BENCHMARK(BM_FFT_ROUNDTRIP_CPUIPP_MEM)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 15)
    ->Complexity();

BENCHMARK_MAIN();
