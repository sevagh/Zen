#include "hpss_cpu.h"
#include "rhythm_toolkit/hpss.h"
#include "util.h"
#include <benchmark/benchmark.h>

static void BM_HPSS_GPUCUDA(benchmark::State& state)
{
	auto data = test_util::sinewave(state.range(0), 1337, 48000);
	for (auto _ : state) {
		std::cout << "hello world" << std::endl;
		// code to measure goes here
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_HPSS_GPUCUDA)->Range(1 << 10, 1 << 20)->Complexity();
BENCHMARK_MAIN();
