#include <algorithm>
#include <functional>
#include <iostream>

#include "rhythm_toolkit/hpss.h"

int
main(int argc, char **argv)
{
	auto hpss = rhythm_toolkit::hpss::HPSS(48000, 8, 16, 4, 2);
	std::vector<std::vector<float>> vals{{1, 2, 3, 4}, {5, 6, 7, 8}};

	for (int i = 0; i < 2; i++) {
		hpss.process_next_hop(vals[i]);
	}

	return 0;
}
