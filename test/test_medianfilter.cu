#include <gtest/gtest.h>
#include "medianfilter.h"
#include "rhythm_toolkit/rhythm_toolkit.h"
#include <iostream>

#include <thrust/device_vector.h>
#include "npp.h"
#include "nppdefs.h"
#include "nppi.h"

using namespace rhythm_toolkit_private::median_filter;

  class MedianFilterTest : public ::testing::Test {

  public:
	  thrust::device_vector<float> _testdata;
	  thrust::device_vector<float> _result;

    Npp32f *testdata;
    Npp32f *result;
    MedianFilterGPU* time_mfilt;
    MedianFilterGPU* freq_mfilt;
    int x;
    int y;

    MedianFilterTest(int x, int y, int f)
    : x(x)
    , y(y)
	, _testdata(thrust::device_vector<float>(x*y))
	, _result(thrust::device_vector<float>(x*y))
    {
	testdata = (Npp32f*)thrust::raw_pointer_cast(_testdata.data());
	result = (Npp32f*)thrust::raw_pointer_cast(_result.data());

	if (testdata == nullptr) {
		std::cerr << "couldn't allocate device memory for test vectors" << std::endl;
		std::exit(1);
	}

	// fill middle row and middle column
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			if (i == x/2)
				_testdata[i*y + j] = 5;
			if (j == y/2)
				_testdata[i*y + j] = 8;
		}
	}

        time_mfilt = new MedianFilterGPU(x, y, f, MedianFilterDirection::Time);
        freq_mfilt = new MedianFilterGPU(x, y, f, MedianFilterDirection::Frequency);
    }

    virtual ~MedianFilterTest() {
	    delete time_mfilt;
	    delete freq_mfilt;
    }

    virtual void SetUp() {}

    virtual void TearDown() {}
  };

class MedianFilterSmallSquareUnitTest : public MedianFilterTest {
protected:
    MedianFilterSmallSquareUnitTest() : MedianFilterTest(9, 9, 3) {}
};

class MedianFilterLargeSquareUnitTest : public MedianFilterTest {
protected:
    MedianFilterLargeSquareUnitTest() : MedianFilterTest(1024, 1024, 21) {}
};

class MedianFilterSmallRectangleUnitTest : public MedianFilterTest {
protected:
    MedianFilterSmallRectangleUnitTest() : MedianFilterTest(10, 20, 5) {}
};

class MedianFilterLargeRectangleUnitTest : public MedianFilterTest {
protected:
    MedianFilterLargeRectangleUnitTest() : MedianFilterTest(10, 20, 5) {}
};

TEST_F(MedianFilterSmallSquareUnitTest, Time)
{
	std::cout << "before" << std::endl;
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _testdata[i*y + j];
			std::cout << elem << " ";
		}
		std::cout << std::endl;
	}

	time_mfilt->filter(testdata, result);

	std::cout << "after" << std::endl;
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];
			std::cout << elem << " ";
		}
		std::cout << std::endl;
	}

	//for (int i = 0; i < x; ++i) {
	//	for (int j = 0; j < y; ++j) {
	//		auto elem = _result[i*y + j];
	//		if (i == x/2) {
	//			EXPECT_EQ(elem, 5);
	//		} else {
	//			EXPECT_EQ(elem, 0);
	//		}
	//	}
	//}
}

TEST_F(MedianFilterSmallRectangleUnitTest, Time)
{
	std::cout << "before" << std::endl;
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _testdata[i*y + j];
			std::cout << elem << " ";
		}
		std::cout << std::endl;
	}

	time_mfilt->filter(testdata, result);

	std::cout << "after" << std::endl;
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];
			std::cout << elem << " ";
		}
		std::cout << std::endl;
	}

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];

			if (i == x/2) {
				EXPECT_EQ(elem, 5);
			} else {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTest, Time)
{
	time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];

			if (i == x/2) {
				EXPECT_EQ(elem, 5);
			} else {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallSquareUnitTest, Frequency)
{
	freq_mfilt->filter(testdata, result);


	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];

			// allow 0s on the outermost edges from the limited roi
			if (i == x/2 && j >= 2 && j < y-2) {
				EXPECT_EQ(elem, 5);
			} else if (i != x/2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTest, Frequency)
{
	freq_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];

			if (i == x/2 && j > 2 && j < y-3) {
				EXPECT_EQ(elem, 5);
			} else if (i != x/2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTest, Frequency)
{
	freq_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i*y + j];

			if (i == x/2 && j >= 3 && j < y-3) {
				EXPECT_EQ(elem, 5);
			} else if (i != x/2) {
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
