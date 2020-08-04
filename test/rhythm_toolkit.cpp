#include <iostream>
#include <rhythm_toolkit/hpss.h>
#include <rhythm_toolkit/rhythm_toolkit.h>
#include <opencv2/core/utility.hpp>

int main(int argc, char** argv)
{
	rhythm_toolkit::hello_world();
	auto hpss = rhythm_toolkit::hpss::HPSS();
	hpss.do_work();
	std::cout << cv::getBuildInformation() << std::endl;
	return 0;
}
