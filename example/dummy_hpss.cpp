#include <algorithm>
#include <functional>
#include <iostream>
#include <cstddef>

#include "rhythm_toolkit/hpss.h"
#include "ffts/ffts.h"

int
main(int argc, char **argv)
{
	ffts_plan_t *p = ffts_init_1d(8, FFTS_FORWARD);
	std::vector<float> arr{0.01F, 0.03F, 0.05F, 0.06F};
	std::vector<std::complex<float>> result(8);

	for (std::size_t i = 0; i < 4; ++i) {
		result[i] = std::complex<float>(arr[i], 0.0F);
	}
	std::fill(result.begin()+4, result.end(), std::complex<float>(0.0F, 0.0F));

	ffts_execute(p, result.data(), result.data());

	std::cout << "fft result" << std::endl;
	for (std::size_t i = 0; i < 8; ++i) {
		std::cout << result[i] << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	return 0;
}
