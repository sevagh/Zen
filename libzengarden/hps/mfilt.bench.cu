#include <benchmark/benchmark.h>
#include <hps/mfilt.h>
#include <libzengarden/io.h>
#include <thrust/device_vector.h>

// l_harm and l_perc are typically no more than 11
static constexpr int TYPICAL_FILTER_LEN = 11;
static constexpr int MAX_DIM = 16384;

using namespace zg::hps::mfilt;
using namespace zg::io;

// make a global IOGPU object because reinitializing it over and over in every
// benchmark was causing some strange cuda errors
IOGPU global_io = IOGPU(MAX_DIM * MAX_DIM);

static void BM_MEDIANFILTER_HORIZONTAL_GPUCUDA_NOMEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = thrust::device_vector<float>(dim * dim);
	thrust::sequence(data.begin(), data.end(), 0.0F);
	auto result = thrust::device_vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::Frequency);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_HORIZONTAL_GPUCUDA_MEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::Frequency);
	for (auto _ : state) {
		// copy into the mapped host-device input memory
		std::copy(data.begin(), data.end(), global_io.host_in);

		// operate on the device mapped pointer
		mfilt.filter(global_io.device_in, global_io.device_out);

		// copy the median filtered data back out from the mapped output memory
		std::copy(global_io.host_out, global_io.host_out + data.size(),
		          result.begin());
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_VERTICAL_GPUCUDA_NOMEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = thrust::device_vector<float>(dim * dim);
	thrust::sequence(data.begin(), data.end(), 0.0F);
	auto result = thrust::device_vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::TimeCausal);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_VERTICAL_GPUCUDA_MEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::TimeCausal);
	for (auto _ : state) {
		// copy into the mapped host-device input memory
		std::copy(data.begin(), data.end(), global_io.host_in);

		// operate on the device mapped pointer
		mfilt.filter(global_io.device_in, global_io.device_out);

		// copy the median filtered data back out from the mapped output memory
		std::copy(global_io.host_out, global_io.host_out + data.size(),
		          result.begin());
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_HORIZONTAL_CPUIPP(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterCPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::Frequency);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_VERTICAL_CPUIPP(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterCPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::TimeCausal);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void
BM_MEDIANFILTER_HORIZONTAL_GPUCUDACOPYBORD_NOMEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = thrust::device_vector<float>(dim * dim);
	thrust::sequence(data.begin(), data.end(), 0.0F);
	auto result = thrust::device_vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::Frequency, true);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void
BM_MEDIANFILTER_HORIZONTAL_GPUCUDACOPYBORD_MEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::Frequency, true);
	for (auto _ : state) {
		// copy into the mapped host-device input memory
		std::copy(data.begin(), data.end(), global_io.host_in);

		// operate on the device mapped pointer
		mfilt.filter(global_io.device_in, global_io.device_out);

		// copy the median filtered data back out from the mapped output memory
		std::copy(global_io.host_out, global_io.host_out + data.size(),
		          result.begin());
	}
	state.SetComplexityN(state.range(0));
}

static void
BM_MEDIANFILTER_VERTICAL_GPUCUDACOPYBORD_NOMEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = thrust::device_vector<float>(dim * dim);
	thrust::sequence(data.begin(), data.end(), 0.0F);
	auto result = thrust::device_vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::TimeCausal, true);
	for (auto _ : state) {
		mfilt.filter(data, result);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_MEDIANFILTER_VERTICAL_GPUCUDACOPYBORD_MEM(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim * dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim * dim);

	auto mfilt = MedianFilterGPU(
	    dim, dim, TYPICAL_FILTER_LEN, MedianFilterDirection::TimeCausal, true);
	for (auto _ : state) {
		// copy into the mapped host-device input memory
		std::copy(data.begin(), data.end(), global_io.host_in);

		// operate on the device mapped pointer
		mfilt.filter(global_io.device_in, global_io.device_out);

		// copy the median filtered data back out from the mapped output memory
		std::copy(global_io.host_out, global_io.host_out + data.size(),
		          result.begin());
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_MEDIANFILTER_HORIZONTAL_GPUCUDA_NOMEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_HORIZONTAL_GPUCUDA_MEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_HORIZONTAL_GPUCUDACOPYBORD_NOMEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_HORIZONTAL_GPUCUDACOPYBORD_MEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_HORIZONTAL_CPUIPP)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();

BENCHMARK(BM_MEDIANFILTER_VERTICAL_GPUCUDA_NOMEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_VERTICAL_GPUCUDA_MEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_VERTICAL_GPUCUDACOPYBORD_NOMEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_VERTICAL_GPUCUDACOPYBORD_MEM)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();
BENCHMARK(BM_MEDIANFILTER_VERTICAL_CPUIPP)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 14)
    ->Complexity();

BENCHMARK_MAIN();
