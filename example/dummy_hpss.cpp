#include <algorithm>
#include <functional>
#include <iostream>

#include "rhythm_toolkit/hpss.h"

int
main(int argc, char **argv)
{
	auto hpss = rhythm_toolkit::hpss::HPSS(1000.0, 4, 2.0);
	std::vector<std::vector<float>> vals{{10, 2, 3, 4}, {10, 6, 7, 8}, {10, 5, 1337, 0}};

	for (int i = 0; i < 3; i++) {
		hpss.process_next_hop(vals[i]);
		auto perc = hpss.peek_separated_percussive();

		std::cout << "p" << std::endl;
		for (std::size_t i = 0; i < 4; ++i) {
			std::cout << perc[i] << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}

	return 0;
}
