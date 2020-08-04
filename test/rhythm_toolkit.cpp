#include <iostream>
#include <rhythm_toolkit/hpss.h>
#include <rhythm_toolkit/rhythm_toolkit.h>

int main(int argc, char** argv)
{
	rhythm_toolkit::hello_world();
	auto hpss = rhythm_toolkit::hpss::HPSS();
	hpss.do_work();
	return 0;
}
