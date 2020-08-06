#include <iostream>
#include <rhythm_toolkit/hpss.h>
#include <rhythm_toolkit/rhythm_toolkit.h>

int main(int argc, char** argv)
{
	rhythm_toolkit::hello_world();
	auto hpss = rhythm_toolkit::hpss::HPSS(48000.0);
	std::vector<float> fake_data(512);
	hpss.process_current_hop(fake_data);
	return 0;
}
