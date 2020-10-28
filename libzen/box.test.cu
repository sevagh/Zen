#include <box.h>
#include <gtest/gtest.h>
#include <hps.h>
#include <iostream>
#include <mfilt.h>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace zen::internal::hps::mfilt;
using namespace zen::internal::hps::box;
using namespace zen::internal::hps;
using namespace zen;

class BoxFilterGPUTest : public ::testing::Test {

public:
	thrust::device_vector<float> testdata;
	thrust::device_vector<float> result;

	BoxFilterGPU* causal_time_bfilt;
	BoxFilterGPU* anticausal_time_bfilt;
	BoxFilterGPU* freq_bfilt;
	int x;
	int y;
	int f;

	BoxFilterGPUTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , f(f)
	    , testdata(thrust::device_vector<float>(x * y))
	    , result(thrust::device_vector<float>(x * y))
	{
		// fill middle row and middle column
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				if (j == y / 2)
					testdata[i * y + j] = 8;
				else
					testdata[i * y + j] = 0;
			}
		}

		causal_time_bfilt
		    = new BoxFilterGPU(x, y, f, MedianFilterDirection::TimeCausal);
		anticausal_time_bfilt
		    = new BoxFilterGPU(x, y, f, MedianFilterDirection::TimeAnticausal);
		freq_bfilt
		    = new BoxFilterGPU(x, y, f, MedianFilterDirection::Frequency);
	}

	virtual ~BoxFilterGPUTest()
	{
		delete causal_time_bfilt;
		delete anticausal_time_bfilt;
		delete freq_bfilt;
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

class BoxFilterSmallSquareUnitTestGPU : public BoxFilterGPUTest {
protected:
	BoxFilterSmallSquareUnitTestGPU()
	    : BoxFilterGPUTest(9, 9, 3)
	{
	}
};

class BoxFilterLargeSquareUnitTestGPU : public BoxFilterGPUTest {
protected:
	BoxFilterLargeSquareUnitTestGPU()
	    : BoxFilterGPUTest(1024, 1024, 21)
	{
	}
};

class BoxFilterSmallRectangleUnitTestGPU : public BoxFilterGPUTest {
protected:
	BoxFilterSmallRectangleUnitTestGPU()
	    : BoxFilterGPUTest(10, 20, 5)
	{
	}
};

class BoxFilterLargeRectangleUnitTestGPU : public BoxFilterGPUTest {
protected:
	BoxFilterLargeRectangleUnitTestGPU()
	    : BoxFilterGPUTest(1024, 17, 5)
	{
	}
};

TEST_F(BoxFilterSmallSquareUnitTestGPU, CausalTime)
{
	// printPre();

	thrust::transform(testdata.begin(), testdata.end(), testdata.begin(),
	                  zen::internal::hps::reciprocal_functor(1.0F));

	causal_time_bfilt->filter(testdata, result);

	thrust::transform(result.begin(), result.end(), result.begin(),
	                  zen::internal::hps::reciprocal_functor(f + 1.0F));

	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 32);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(BoxFilterSmallRectangleUnitTestGPU, CausalTime)
{
	// printPre();
	thrust::transform(testdata.begin(), testdata.end(), testdata.begin(),
	                  zen::internal::hps::reciprocal_functor(1.0F));

	causal_time_bfilt->filter(testdata, result);

	thrust::transform(result.begin(), result.end(), result.begin(),
	                  zen::internal::hps::reciprocal_functor(f + 1.0F));

	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 48);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(BoxFilterLargeRectangleUnitTestGPU, CausalTime)
{
	// printPre();
	thrust::transform(testdata.begin(), testdata.end(), testdata.begin(),
	                  zen::internal::hps::reciprocal_functor(1.0F));

	causal_time_bfilt->filter(testdata, result);

	thrust::transform(result.begin(), result.end(), result.begin(),
	                  zen::internal::hps::reciprocal_functor(f + 1.0F));

	// printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];
			if (j == y / 2) {
				EXPECT_EQ(elem, 48);
			}
			else if (j != y / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

TEST_F(BoxFilterSmallSquareUnitTestGPU, Frequency)
{
	printPre();

	thrust::transform(testdata.begin(), testdata.end(), testdata.begin(),
	                  zen::internal::hps::reciprocal_functor(1.0F));

	printPre();

	freq_bfilt->filter(testdata, result);

	printPost();

	thrust::transform(result.begin(), result.end(), result.begin(),
	                  zen::internal::hps::reciprocal_functor(f + 1.0F));

	printPost();

	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < y; ++j) {
			auto elem = result[i * y + j];

			// allow 0s on the outermost edges from the limited roi
			if (i == x / 2) {
				EXPECT_EQ(elem, 20);
			}
			else if (i != x / 2) {
				EXPECT_EQ(elem, 0);
			}
		}
	}
}

//TEST_F(BoxFilterSmallRectangleUnitTestGPU, Frequency)
//{
//	// printPre();
//	freq_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//
//			if (i == x / 2 && j < y - 5) {
//				EXPECT_EQ(elem, 5);
//			}
//			else if (i != x / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterLargeRectangleUnitTestGPU, Frequency)
//{
//	// printPre();
//	freq_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//
//			if (i == x / 2 && j < y - 5) {
//				EXPECT_EQ(elem, 5);
//			}
//			else if (i != x / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST(BoxFilterUnitTestGPU, DegenerateInputFilterTooBig)
//{
//	EXPECT_THROW(BoxFilterGPU(9, 9, 171, MedianFilterDirection::Frequency),
//	             ZgException);
//	EXPECT_THROW(BoxFilterGPU(9, 9, 171, MedianFilterDirection::TimeCausal),
//	             ZgException);
//	EXPECT_THROW(BoxFilterGPU(9, 9, 171, MedianFilterDirection::TimeAnticausal),
//	             ZgException);
//}
//
//TEST_F(BoxFilterSmallSquareUnitTestGPU, AnticausalTime)
//{
//	printPre();
//
//	thrust::transform(testdata.begin(), testdata.end(), testdata.begin(),
//	                  zen::internal::hps::reciprocal_functor(1.0F));
//
//	anticausal_time_bfilt->filter(testdata, result);
//
//	thrust::transform(result.begin(), result.end(), result.begin(),
//	                  zen::internal::hps::reciprocal_functor(f + 1.0F));
//
//	printPost();
//
//	//for (int i = 0; i < x; ++i) {
//	//	for (int j = 0; j < y; ++j) {
//	//		auto elem = result[i * y + j];
//	//		if (j == y / 2 && i > 2 && i < x - 3) {
//	//			EXPECT_EQ(elem, 8);
//	//		}
//	//		else if (j != y / 2) {
//	//			EXPECT_EQ(elem, 0);
//	//		}
//	//	}
//	//}
//}
//
//TEST_F(BoxFilterSmallRectangleUnitTestGPU, AnticausalTime)
//{
//	// printPre();
//	anticausal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 2 && i < x - 3) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterLargeRectangleUnitTestGPU, AnticausalTime)
//{
//	// printPre();
//	anticausal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 2 && i < x - 3) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}

class BoxFilterCPUTest : public ::testing::Test {

public:
	std::vector<float> testdata;
	std::vector<float> result;

	BoxFilterCPU* causal_time_bfilt;
	BoxFilterCPU* anticausal_time_bfilt;
	BoxFilterCPU* freq_bfilt;
	int x;
	int y;
	int f;

	BoxFilterCPUTest(int x, int y, int f)
	    : x(x)
	    , y(y)
	    , f(f)
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

		causal_time_bfilt
		    = new BoxFilterCPU(x, y, f, MedianFilterDirection::TimeCausal);
		anticausal_time_bfilt
		    = new BoxFilterCPU(x, y, f, MedianFilterDirection::TimeAnticausal);
		freq_bfilt
		    = new BoxFilterCPU(x, y, f, MedianFilterDirection::Frequency);
	}

	virtual ~BoxFilterCPUTest()
	{
		delete causal_time_bfilt;
		delete anticausal_time_bfilt;
		delete freq_bfilt;
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

class BoxFilterSmallSquareUnitTestCPU : public BoxFilterCPUTest {
protected:
	BoxFilterSmallSquareUnitTestCPU()
	    : BoxFilterCPUTest(9, 9, 3)
	{
	}
};

class BoxFilterLargeSquareUnitTestCPU : public BoxFilterCPUTest {
protected:
	BoxFilterLargeSquareUnitTestCPU()
	    : BoxFilterCPUTest(1024, 1024, 21)
	{
	}
};

class BoxFilterSmallRectangleUnitTestCPU : public BoxFilterCPUTest {
protected:
	BoxFilterSmallRectangleUnitTestCPU()
	    : BoxFilterCPUTest(10, 20, 5)
	{
	}
};

class BoxFilterLargeRectangleUnitTestCPU : public BoxFilterCPUTest {
protected:
	BoxFilterLargeRectangleUnitTestCPU()
	    : BoxFilterCPUTest(1024, 128, 5)
	{
	}
};

TEST_F(BoxFilterSmallSquareUnitTestCPU, CausalTime)
{
	printPre();

	std::transform(testdata.begin(), testdata.end(), testdata.begin(),
	               zen::internal::hps::reciprocal_functor(1.0F));

	printPre();

	causal_time_bfilt->filter(testdata, result);

	printPost();

	std::transform(result.begin(), result.end(), result.begin(),
	               zen::internal::hps::reciprocal_functor(f + 1.0F));

	printPost();

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

//TEST_F(BoxFilterSmallRectangleUnitTestCPU, CausalTime)
//{
//	// printPre();
//	causal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 5) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterLargeRectangleUnitTestCPU, CausalTime)
//{
//	// printPre();
//	causal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 5) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterSmallSquareUnitTestCPU, Frequency)
//{
//	// printPre();
//	freq_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//
//			// allow 0s on the outermost edges from the limited roi
//			if (i == x / 2 && j < y - 3) {
//				EXPECT_EQ(elem, 5);
//			}
//			else if (i != x / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterSmallRectangleUnitTestCPU, Frequency)
//{
//	// printPre();
//	freq_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//
//			if (i == x / 2 && j < y - 5) {
//				EXPECT_EQ(elem, 5);
//			}
//			else if (i != x / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterLargeRectangleUnitTestCPU, Frequency)
//{
//	// printPre();
//	freq_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//
//			if (i == x / 2 && j < y - 5) {
//				EXPECT_EQ(elem, 5);
//			}
//			else if (i != x / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST(BoxFilterUnitTestCPU, DegenerateInputFilterTooBig)
//{
//	EXPECT_THROW(BoxFilterCPU(9, 9, 171, MedianFilterDirection::Frequency),
//	             ZgException);
//	EXPECT_THROW(BoxFilterCPU(9, 9, 171, MedianFilterDirection::TimeCausal),
//	             ZgException);
//	EXPECT_THROW(BoxFilterCPU(9, 9, 171, MedianFilterDirection::TimeAnticausal),
//	             ZgException);
//}
//
//TEST_F(BoxFilterSmallSquareUnitTestCPU, AnticausalTime)
//{
//	// printPre();
//	anticausal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 2 && i < x - 3) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterSmallRectangleUnitTestCPU, AnticausalTime)
//{
//	// printPre();
//	anticausal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 2 && i < x - 3) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
//
//TEST_F(BoxFilterLargeRectangleUnitTestCPU, AnticausalTime)
//{
//	// printPre();
//	anticausal_time_bfilt->filter(testdata, result);
//	// printPost();
//
//	for (int i = 0; i < x; ++i) {
//		for (int j = 0; j < y; ++j) {
//			auto elem = result[i * y + j];
//			if (j == y / 2 && i > 2 && i < x - 3) {
//				EXPECT_EQ(elem, 8);
//			}
//			else if (j != y / 2) {
//				EXPECT_EQ(elem, 0);
//			}
//		}
//	}
//}
