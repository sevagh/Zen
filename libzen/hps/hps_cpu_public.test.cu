#include <gtest/gtest.h>
#include <iostream>
#include <libzen/hps.h>
#include <libzen/io.h>
#include <random>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

#include <cmath>

using namespace zen::hps;
using namespace zen;

static std::vector<float> generate_data_normalized(size_t size)
{
	// use realistic values of normalized input floats
	static std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
	static std::default_random_engine generator;

	std::vector<float> data(size);

	std::generate(
	    data.begin(), data.end(), []() { return distribution(generator); });

	return data;
}

zen::io::IOGPU io_obj(8192);

// 20x 4096 data
class HPRIOfflineCPUTest : public ::testing::Test {
public:
	static constexpr std::size_t big_hop = 4096;
	static constexpr std::size_t small_hop = 256;
	std::size_t n_big_hops;
	std::size_t n_small_hops;

	HPRIOffline<Backend::CPU> hpri_offline;
	PRealtime<Backend::CPU> p_rt;

	std::vector<float> testdata;
	std::vector<float> prt_result;

	HPRIOfflineCPUTest()
	    : n_big_hops(20)
	    , hpri_offline(48000.0F, big_hop, small_hop, 2.0, 2.0)
	    , p_rt(48000.0F, small_hop, 2.0)
	    , testdata(generate_data_normalized(n_big_hops * big_hop))
	    , prt_result(n_big_hops * big_hop)
	{
		n_small_hops = (std::size_t)(
		    (( float )n_big_hops) * (( float )big_hop / ( float )small_hop));
	};

	virtual void SetUp() {}

	virtual void TearDown() {}
};

TEST_F(HPRIOfflineCPUTest, Basic)
{
	auto ret = hpri_offline.process(testdata);
	EXPECT_EQ(ret[1].size(), testdata.size());

	for (std::size_t i = 0; i < big_hop * n_big_hops; ++i) {
		EXPECT_NE(ret[1][i], testdata[i]);
	}
	for (std::size_t i = 0; i < n_small_hops; ++i) {
		p_rt.process_next_hop(testdata.data() + i * small_hop,
		                      prt_result.data() + i * small_hop);
		for (std::size_t j = 0; j < small_hop; ++j) {
			EXPECT_NE(
			    testdata[i * small_hop + j], prt_result[i * small_hop + j]);
		}
	}
}

TEST_F(HPRIOfflineCPUTest, WithPadding)
{
	auto oldsize = testdata.size();
	testdata.resize(testdata.size() + 11);

	auto ret = hpri_offline.process(testdata);
	EXPECT_EQ(ret[1].size(), testdata.size());
	EXPECT_NE(ret[1].size(), oldsize);

	for (std::size_t i = 0; i < big_hop * n_big_hops; ++i) {
		EXPECT_NE(ret[1][i], testdata[i]);
	}
	for (std::size_t i = 0; i < n_small_hops; ++i) {
		p_rt.process_next_hop(testdata.data() + i * small_hop,
		                      prt_result.data() + i * small_hop);
		for (std::size_t j = 0; j < small_hop; ++j) {
			EXPECT_NE(
			    testdata[i * small_hop + j], prt_result[i * small_hop + j]);
		}
	}
}
