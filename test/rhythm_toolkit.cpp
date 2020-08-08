#include <iostream>
#include <numeric>
#include <rhythm_toolkit/hpss.h>
#include <rhythm_toolkit/rhythm_toolkit.h>

int main(int argc, char** argv)
{
	rhythm_toolkit::hello_world();
	auto hpss = rhythm_toolkit::hpss::HPSS(48000.0);
	std::vector<float> fake_data(512);
	std::iota(fake_data.begin(), fake_data.end(), 0);
	hpss.process_next_hop(fake_data);
	return 0;
}
