#include <gtest/gtest.h>
#include <hps/mfilt.h>
#include <iostream>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

using namespace zg::internal::hps::mfilt;
using namespace zg;

class MedianFilterGPUTest : public ::testing::Test {

public:
	thrust::device_vector<float> testdata;
	thrust::device_vector<float> result;

	MedianFilterGPU* causal_time_mfilt;
	MedianFilterGPU* anticausal_time_mfilt;
	MedianFilterGPU* freq_mfilt;
	int x;
	int y;

	MedianFilterGPUTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , testdata(thrust::device_vector<float>(x * y))
	    , result(thrust::device_vector<float>(x * y))
	{
		// fill middle row and middle column
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				if (i == x / 2)
					testdata[i * y + j] = 5;
				if (j == y / 2)
					testdata[i * y + j] = 8;
			}
		}

		causal_time_mfilt
		    = new MedianFilterGPU(x, y, f, MedianFilterDirection::TimeCausal);
		anticausal_time_mfilt = new MedianFilterGPU(
		    x, y, f, MedianFilterDirection::TimeAnticausal);
		freq_mfilt
		    = new MedianFilterGPU(x, y, f, MedianFilterDirection::Frequency);
	}

	virtual ~MedianFilterGPUTest()
	{
		delete causal_time_mfilt;
		delete anticausal_time_mfilt;
		delete freq_mfilt;
	}

	void printPre()
	{
		std::cout << "before" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = testdata[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	void printPost()
	{
		std::cout << "after" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = result[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class MedianFilterSmallSquareUnitTestGPU : public MedianFilterGPUTest {
protected:
	MedianFilterSmallSquareUnitTestGPU()
	    : MedianFilterGPUTest(9, 9, 3)
	{
	}
};

class MedianFilterLargeSquareUnitTestGPU : public MedianFilterGPUTest {
protected:
	MedianFilterLargeSquareUnitTestGPU()
	    : MedianFilterGPUTest(1024, 1024, 21)
	{
	}
};

class MedianFilterSmallRectangleUnitTestGPU : public MedianFilterGPUTest {
protected:
	MedianFilterSmallRectangleUnitTestGPU()
	    : MedianFilterGPUTest(10, 20, 5)
	{
	}
};

class MedianFilterLargeRectangleUnitTestGPU : public MedianFilterGPUTest {
protected:
	MedianFilterLargeRectangleUnitTestGPU()
	    : MedianFilterGPUTest(1024, 17, 5)
	{
	}
};

TEST_F(MedianFilterSmallSquareUnitTestGPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestGPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallSquareUnitTestGPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

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

TEST_F(MedianFilterSmallRectangleUnitTestGPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTestGPU, DegenerateInputFilterTooBig)
{
	EXPECT_THROW(MedianFilterGPU(9, 9, 171, MedianFilterDirection::Frequency),
	             ZgException);
	EXPECT_THROW(MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeCausal),
	             ZgException);
	EXPECT_THROW(
	    MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeAnticausal),
	    ZgException);
}

TEST_F(MedianFilterSmallSquareUnitTestGPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestGPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

class MedianFilterCPUTest : public ::testing::Test {

public:
	std::vector<float> testdata;
	std::vector<float> result;

	MedianFilterCPU* causal_time_mfilt;
	MedianFilterCPU* anticausal_time_mfilt;
	MedianFilterCPU* freq_mfilt;
	int x;
	int y;

	MedianFilterCPUTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , testdata(std::vector<float>(x * y))
	    , result(std::vector<float>(x * y))
	{
		// fill middle row and middle column
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				if (i == x / 2)
					testdata[i * y + j] = 5;
				if (j == y / 2)
					testdata[i * y + j] = 8;
			}
		}

		causal_time_mfilt
		    = new MedianFilterCPU(x, y, f, MedianFilterDirection::TimeCausal);
		anticausal_time_mfilt = new MedianFilterCPU(
		    x, y, f, MedianFilterDirection::TimeAnticausal);
		freq_mfilt
		    = new MedianFilterCPU(x, y, f, MedianFilterDirection::Frequency);
	}

	virtual ~MedianFilterCPUTest()
	{
		delete causal_time_mfilt;
		delete anticausal_time_mfilt;
		delete freq_mfilt;
	}

	void printPre()
	{
		std::cout << "before" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = testdata[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	void printPost()
	{
		std::cout << "after" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = result[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class MedianFilterSmallSquareUnitTestCPU : public MedianFilterCPUTest {
protected:
	MedianFilterSmallSquareUnitTestCPU()
	    : MedianFilterCPUTest(9, 9, 3)
	{
	}
};

class MedianFilterLargeSquareUnitTestCPU : public MedianFilterCPUTest {
protected:
	MedianFilterLargeSquareUnitTestCPU()
	    : MedianFilterCPUTest(1024, 1024, 21)
	{
	}
};

class MedianFilterSmallRectangleUnitTestCPU : public MedianFilterCPUTest {
protected:
	MedianFilterSmallRectangleUnitTestCPU()
	    : MedianFilterCPUTest(10, 20, 5)
	{
	}
};

class MedianFilterLargeRectangleUnitTestCPU : public MedianFilterCPUTest {
protected:
	MedianFilterLargeRectangleUnitTestCPU()
	    : MedianFilterCPUTest(1024, 128, 5)
	{
	}
};

TEST_F(MedianFilterSmallSquareUnitTestCPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestCPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestCPU, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 5) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallSquareUnitTestCPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

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

TEST_F(MedianFilterSmallRectangleUnitTestCPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestCPU, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2 && j < y - 5) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTestCPU, DegenerateInputFilterTooBig)
{
	EXPECT_THROW(MedianFilterCPU(9, 9, 171, MedianFilterDirection::Frequency),
	             ZgException);
	EXPECT_THROW(MedianFilterCPU(9, 9, 171, MedianFilterDirection::TimeCausal),
	             ZgException);
	EXPECT_THROW(
	    MedianFilterCPU(9, 9, 171, MedianFilterDirection::TimeAnticausal),
	    ZgException);
}

TEST_F(MedianFilterSmallSquareUnitTestCPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestCPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestCPU, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2 && i > 2 && i < x - 3) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

class MedianFilterGPUCOPYBORDTest : public ::testing::Test {

public:
	thrust::device_vector<float> testdata;
	thrust::device_vector<float> result;

	MedianFilterGPU* causal_time_mfilt;
	MedianFilterGPU* anticausal_time_mfilt;
	MedianFilterGPU* freq_mfilt;
	int x;
	int y;

	MedianFilterGPUCOPYBORDTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , testdata(thrust::device_vector<float>(x * y))
	    , result(thrust::device_vector<float>(x * y))
	{
		// fill middle row and middle column
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				if (i == x / 2)
					testdata[i * y + j] = 5;
				if (j == y / 2)
					testdata[i * y + j] = 8;
			}
		}

		causal_time_mfilt = new MedianFilterGPU(
		    x, y, f, MedianFilterDirection::TimeCausal, true);
		anticausal_time_mfilt = new MedianFilterGPU(
		    x, y, f, MedianFilterDirection::TimeAnticausal, true);
		freq_mfilt = new MedianFilterGPU(
		    x, y, f, MedianFilterDirection::Frequency, true);
	}

	virtual ~MedianFilterGPUCOPYBORDTest()
	{
		delete causal_time_mfilt;
		delete anticausal_time_mfilt;
		delete freq_mfilt;
	}

	void printPre()
	{
		std::cout << "before" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = testdata[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	void printPost()
	{
		std::cout << "after" << std::endl;
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				auto elem = result[i * y + j];
				std::cout << elem << " ";
			}
			std::cout << std::endl;
		}
	}

	virtual void SetUp() {}

	virtual void TearDown() {}
};

class MedianFilterSmallSquareUnitTestGPUCOPYBORD
    : public MedianFilterGPUCOPYBORDTest {
protected:
	MedianFilterSmallSquareUnitTestGPUCOPYBORD()
	    : MedianFilterGPUCOPYBORDTest(9, 9, 3)
	{
	}
};

class MedianFilterLargeSquareUnitTestGPUCOPYBORD
    : public MedianFilterGPUCOPYBORDTest {
protected:
	MedianFilterLargeSquareUnitTestGPUCOPYBORD()
	    : MedianFilterGPUCOPYBORDTest(1024, 1024, 21)
	{
	}
};

class MedianFilterSmallRectangleUnitTestGPUCOPYBORD
    : public MedianFilterGPUCOPYBORDTest {
protected:
	MedianFilterSmallRectangleUnitTestGPUCOPYBORD()
	    : MedianFilterGPUCOPYBORDTest(10, 20, 5)
	{
	}
};

class MedianFilterLargeRectangleUnitTestGPUCOPYBORD
    : public MedianFilterGPUCOPYBORDTest {
protected:
	MedianFilterLargeRectangleUnitTestGPUCOPYBORD()
	    : MedianFilterGPUCOPYBORDTest(1024, 17, 5)
	{
	}
};

TEST_F(MedianFilterSmallSquareUnitTestGPUCOPYBORD, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestGPUCOPYBORD, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPUCOPYBORD, CausalTime)
{
	// printPre();
	causal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallSquareUnitTestGPUCOPYBORD, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestGPUCOPYBORD, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPUCOPYBORD, Frequency)
{
	// printPre();
	freq_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			if (i == x / 2) {
				EXPECT_EQ(elem, 5);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST(MedianFilterUnitTestGPUCOPYBORD, DegenerateInputFilterTooBig)
{
	EXPECT_THROW(
	    MedianFilterGPU(9, 9, 171, MedianFilterDirection::Frequency, true),
	    ZgException);
	EXPECT_THROW(
	    MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeCausal, true),
	    ZgException);
	EXPECT_THROW(
	    MedianFilterGPU(9, 9, 171, MedianFilterDirection::TimeAnticausal, true),
	    ZgException);
}

TEST_F(MedianFilterSmallSquareUnitTestGPUCOPYBORD, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterSmallRectangleUnitTestGPUCOPYBORD, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(MedianFilterLargeRectangleUnitTestGPUCOPYBORD, AnticausalTime)
{
	// printPre();
	anticausal_time_mfilt->filter(testdata, result);
	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 8);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}
