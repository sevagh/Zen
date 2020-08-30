#include "medianfilter.h"
#include "rhythm_toolkit/rhythm_toolkit.h"
#include <gtest/gtest.h>
#include <iostream>

#include "npp.h"
#include "nppdefs.h"
#include "nppi.h"
#include <thrust/device_vector.h>

using namespace rhythm_toolkit_private::median_filter;

class MedianFilterTest : public ::testing::Test {

public:
	thrust::device_vector<float> _testdata;
	thrust::device_vector<float> _result;

	Npp32f* testdata;
	Npp32f* result;
	MedianFilterGPU* causal_time_mfilt;
	MedianFilterGPU* anticausal_time_mfilt;
	MedianFilterGPU* freq_mfilt;
	int x;
	int y;

	MedianFilterTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , _testdata(thrust::device_vector<float>(x * y))
	    , _result(thrust::device_vector<float>(x * y))
	{
		testdata = ( Npp32f* )thrust::raw_pointer_cast(_testdata.data());
		result = ( Npp32f* )thrust::raw_pointer_cast(_result.data());

		if (testdata == nullptr) {
			std::cerr << "couldn't allocate device memory for test vectors"
			          << std::endl;
			std::exit(1);
		}

		// fill middle row and middle column
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				if (i == x / 2)
					_testdata[i * y + j] = 5;
				if (j == y / 2)
					_testdata[i * y + j] = 8;
			}
		}

		causal_time_mfilt = new MedianFilterGPU(x, y, f, MedianFilterDirection::TimeCausal);
		anticausal_time_mfilt
		    = new MedianFilterGPU(x, y, f, MedianFilterDirection::TimeAnticausal);
		freq_mfilt
		    = new MedianFilterGPU(x, y, f, MedianFilterDirection::Frequency);
	}

	virtual ~MedianFilterTest()
	{
		delete causal_time_mfilt;
		delete anticausal_time_mfilt;
		delete freq_mfilt;
	}

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class MedianFilterSmallSquareUnitTest : public MedianFilterTest {
protected:
	MedianFilterSmallSquareUnitTest()
	    : MedianFilterTest(9, 9, 3)
	{
	}
};

class MedianFilterLargeSquareUnitTest : public MedianFilterTest {
protected:
	MedianFilterLargeSquareUnitTest()
	    : MedianFilterTest(1024, 1024, 21)
	{
	}
};

class MedianFilterSmallRectangleUnitTest : public MedianFilterTest {
protected:
	MedianFilterSmallRectangleUnitTest()
	    : MedianFilterTest(10, 20, 5)
	{
	}
};

class MedianFilterLargeRectangleUnitTest : public MedianFilterTest {
protected:
	MedianFilterLargeRectangleUnitTest()
	    : MedianFilterTest(10, 20, 5)
	{
	}
};

TEST_F(MedianFilterSmallSquareUnitTest, CausalTime)
{
	causal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTest, CausalTime)
{
	causal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTest, CausalTime)
{
	causal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
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
			auto elem = _result[i * y + j];

			// allow 0s on the outermost edges from the limited roi
			if (i == x / 2 && j < y - 3) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
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
			auto elem = _result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
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
			auto elem = _result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTest, DegenerateInputFilterTooBig)
{
	EXPECT_THROW(MedianFilterGPU(9, 9, 171, MedianFilterDirection::Frequency),
	             rhythm_toolkit::RtkException);
	EXPECT_THROW(MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeCausal),
	             rhythm_toolkit::RtkException);
	EXPECT_THROW(MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeAnticausal),
	             rhythm_toolkit::RtkException);
}

TEST_F(MedianFilterSmallSquareUnitTest, AnticausalTime)
{
	anticausal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTest, AnticausalTime)
{
	anticausal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTest, AnticausalTime)
{
	anticausal_time_mfilt->filter(testdata, result);

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = _result[i * y + j];
			if (j == y / 2 && i < x - 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}
