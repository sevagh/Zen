#include <algorithm>
#include <benchmark/benchmark.h>
#include <libzen/hps.h>
#include <libzen/io.h>
#include <mfilt.h>
#include <numeric>
#include <thrust/device_vector.h>

static constexpr int MAX_DIM = 4096;
static constexpr int SAMPLE_RATE = 48000;

using namespace zen::hps;
using namespace zen;
using namespace zen::io;

// make a global IOGPU object because reinitializing it over and over in every
// benchmark was causing some strange cuda errors
IOGPU global_io = IOGPU(MAX_DIM * MAX_DIM);

static void BM_HPR_GPUCUDA(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim);
	thrust::sequence(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim);

	auto hpr
	    = HPRRealtime<Backend::GPU>(SAMPLE_RATE, dim, 2.0, OUTPUT_PERCUSSIVE);

	for (auto _ : state) {
		std::copy(data.begin(), data.end(), global_io.host_in);

		hpr.process_next_hop(global_io.device_in);
		hpr.copy_percussive(global_io.device_out);

		std::copy(global_io.host_out, global_io.host_out + dim, result.begin());
	}
	state.SetComplexityN(state.range(0));
}

static void BM_HPR_CPU(benchmark::State& state)
{
	// NxN square
	auto dim = state.range(0);

	auto data = std::vector<float>(dim);
	std::iota(data.begin(), data.end(), 0.0F);
	auto result = std::vector<float>(dim);

	auto hpr
	    = HPRRealtime<Backend::CPU>(SAMPLE_RATE, dim, 2.0, OUTPUT_PERCUSSIVE);

	for (auto _ : state) {
		hpr.process_next_hop(data.data());
		hpr.copy_percussive(result.data());
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_HPR_GPUCUDA)->RangeMultiplier(2)->Range(1 << 5, 1 << 12)->Complexity();

BENCHMARK(BM_HPR_CPU)->RangeMultiplier(2)->Range(1 << 5, 1 << 12)->Complexity();

BENCHMARK_MAIN();
