#include "rhythm_toolkit/hpss.h"
#include <iostream>

// uses a locally vendored copy of https://github.com/suomela/mf2d/blob/master/src/filter.h
#include "filter.h"

void rhythm_toolkit::hpss::HPSS::process_current_hop(const std::vector<float> &current_hop)
{
	std::cout << "hpss doing work" << std::endl;

	// do ipp ifft here
}
