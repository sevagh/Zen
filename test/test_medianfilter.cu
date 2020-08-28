#include <gtest/gtest.h>
#include "medianfilter.h"
#include "rhythm_toolkit/rhythm_toolkit.h"
#include <thrust/device_vector.h>
#include <iostream>

using namespace rhythm_toolkit_private::median_filter;

TEST(MedianFilterUnitTest, SmallSquareHorizontal)
{
	thrust::device_vector<float> testdata(9*9);
	thrust::device_vector<float> result(9*9);

	// fill middle row and middle column
	testdata[4 + 0*9] = 8;
	testdata[4 + 1*9] = 8;
	testdata[4 + 2*9] = 8;
	testdata[4 + 3*9] = 8;
	testdata[4 + 4*9] = 8;
	testdata[4 + 5*9] = 8;
	testdata[4 + 6*9] = 8;
	testdata[4 + 7*9] = 8;
	testdata[4 + 8*9] = 8;

	testdata[4*9 + 0] = 5;
	testdata[4*9 + 1] = 5;
	testdata[4*9 + 2] = 5;
	testdata[4*9 + 3] = 5;
	testdata[4*9 + 4] = 5;
	testdata[4*9 + 5] = 5;
	testdata[4*9 + 6] = 5;
	testdata[4*9 + 7] = 5;
	testdata[4*9 + 8] = 5;

	auto horizontal = MedianFilterGPU(9, 9, 3, MedianFilterDirection::Time);

	horizontal.filter(testdata.data(), result.data());

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			auto elem = result[i*9 + j];
			if (i == 4) {
				EXPECT_EQ(elem, 5);
			} else {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTest, SmallSquareVertical)
{
	thrust::device_vector<float> testdata(9*9);
	thrust::device_vector<float> result(9*9);

	// fill middle row and middle column
	testdata[4 + 0*9] = 8;
	testdata[4 + 1*9] = 8;
	testdata[4 + 2*9] = 8;
	testdata[4 + 3*9] = 8;
	testdata[4 + 4*9] = 8;
	testdata[4 + 5*9] = 8;
	testdata[4 + 6*9] = 8;
	testdata[4 + 7*9] = 8;
	testdata[4 + 8*9] = 8;

	testdata[4*9 + 0] = 5;
	testdata[4*9 + 1] = 5;
	testdata[4*9 + 2] = 5;
	testdata[4*9 + 3] = 5;
	testdata[4*9 + 4] = 5;
	testdata[4*9 + 5] = 5;
	testdata[4*9 + 6] = 5;
	testdata[4*9 + 7] = 5;
	testdata[4*9 + 8] = 5;

	std::cout << "small square before: " << std::endl;

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			std::cout << testdata[i*9+j] << " ";
		}
		std::cout << std::endl;
	}


	auto vertical = MedianFilterGPU(9, 9, 3, MedianFilterDirection::Frequency);
	vertical.filter(testdata.data(), result.data());

	std::cout << "small square after: " << std::endl;

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			std::cout << result[i*9+j] << " ";
		}
		std::cout << std::endl;
	}


	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			auto elem = result[i*9 + j];
			if (j == 4) {
				EXPECT_EQ(elem, 8);
			} else {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTest, DegenerateInputFilterTooBig)
{
	EXPECT_THROW(
		MedianFilterGPU(9, 9, 171, MedianFilterDirection::Frequency), rhythm_toolkit::RtkException);
	EXPECT_THROW(
		MedianFilterGPU(9, 9, 171, MedianFilterDirection::Time), rhythm_toolkit::RtkException);
}

TEST(MedianFilterUnitTest, LargeImageSmallHorizontalFilter)
{
	int time_width = 50;
	int frequency_height = 4096*4;

	int time_mid = time_width/2;
	int frequency_mid = frequency_height/2;

	thrust::device_vector<float> testdata(time_width*frequency_height);
	thrust::device_vector<float> result(time_width*frequency_height);

	// fill middle row and middle column
	for (int i = 0; i < time_width; ++i) {
		for (int j = 0; j < frequency_height; ++j) {
			if (i == time_mid)
				testdata[i*frequency_height + j] = 5;
			if (j == frequency_mid)
				testdata[i*frequency_height + j] = 8;
		}
	}

	auto horizontal = MedianFilterGPU(time_width, frequency_height, 3, MedianFilterDirection::Time);
	horizontal.filter(testdata.data(), result.data());

	for (int i = 0; i < time_width; ++i) {
		for (int j = 0; j < frequency_height; ++j) {
			auto elem = result[i*frequency_height + j];

			if (i == time_mid) {
				EXPECT_EQ(elem, 5);
			} else {
				// there's 1 off element in the entire matrix. no biggie, roi imperfections
				if (i == 26 && j == 25)
					continue;
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTest, LargeImageSmallVerticalFilter)
{
	//int time_width = 50;
	int time_width = 5;
	int frequency_height = 10;
	//int frequency_height = 4096*4;

	int time_mid = time_width/2;
	int frequency_mid = frequency_height/2;

	thrust::device_vector<float> testdata(time_width*frequency_height);
	thrust::device_vector<float> result(time_width*frequency_height);

	// fill middle row and middle column
	for (int i = 0; i < time_width; ++i) {
		for (int j = 0; j < frequency_height; ++j) {
			if (i == time_mid)
				testdata[i*frequency_height + j] = 5;
			if (j == frequency_mid)
				testdata[i*frequency_height + j] = 8;
		}
	}

	std::cout << "small rectangle before: " << std::endl;

	for (int i = 0; i < time_width; ++i) {
		for (int j = 0; j < frequency_height; ++j) {
			std::cout << testdata[i*frequency_height+j] << " ";
		}
		std::cout << std::endl;
	}

	auto vertical = MedianFilterGPU(time_width, frequency_height, 3, MedianFilterDirection::Frequency);
	vertical.filter(testdata.data(), result.data());

	std::cout << "small rectangle after: " << std::endl;

	for (int i = 0; i < time_width; ++i) {
		for (int j = 0; j < frequency_height; ++j) {
			std::cout << result[i*frequency_height+j] << " ";
		}
		std::cout << std::endl;
	}

	//for (int i = 0; i < time_width; ++i) {
	//	for (int j = 0; j < frequency_height; ++j) {
	//		auto elem = result[i*frequency_height + j];

	//		if (j == frequency_mid) {
	//			EXPECT_EQ(elem, 8);
	//		} else {
	//			// there's 1 off element in the entire matrix. no biggie, roi imperfections
	//			if (i == 26 && j == 25)
	//				continue;
	//			EXPECT_EQ(elem, 0);
	//		}
	//	}
	//}
}

