#include "rhythm_toolkit/hpss.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

void rhythm_toolkit::hpss::HPSS::process_current_hop(const std::vector<float> &current_hop)
{
	std::cout << "hpss doing work" << std::endl;

	// do ipp fft here


	s_half_mag = cv::abs(sliding_stft);
	cv::medianBlur(s_half_mag, percussive, l_perc);

	// do ipp ifft here
}
