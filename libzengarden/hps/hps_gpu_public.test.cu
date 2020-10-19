#include <gtest/gtest.h>
#include <iostream>
#include <libzengarden/hps.h>
#include <libzengarden/io.h>
#include <random>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

#include <cmath>

using namespace zg::hps;
using namespace zg;

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

zg::io::IOGPU io_obj(8192);

// 20 iterations should be enough to validate correct operation during a whole song?
// anything longer is hell to run with cuda-memcheck
class HPRIOfflineGPUTest : public ::testing::Test {
public:
	static constexpr std::size_t big_hop = 4096;
	static constexpr std::size_t small_hop = 256;
	std::size_t n_big_hops;
	std::size_t n_small_hops;

	HPRIOffline<Backend::GPU> hpri_offline;
	PRealtime<Backend::GPU> p_rt;

	std::vector<float> testdata;

	HPRIOfflineGPUTest()
	    : n_big_hops(20)
	    , hpri_offline(48000.0F, big_hop, small_hop, 2.0, 2.0)
	    , p_rt(48000.0F, small_hop, 2.0)
	    , testdata(generate_data_normalized(n_big_hops * big_hop))
	{
		n_small_hops = (std::size_t)(
		    (( float )n_big_hops) * (( float )big_hop / ( float )small_hop));
	};

	virtual void SetUp() {}

	virtual void TearDown() {}
};

TEST_F(HPRIOfflineGPUTest, Basic)
{
	auto ret = hpri_offline.process(testdata);
	EXPECT_EQ(ret.size(), testdata.size());

	for (std::size_t i = 0; i < big_hop * n_big_hops; ++i) {
		EXPECT_NE(ret[i], testdata[i]);
	}
	for (std::size_t i = 0; i < n_small_hops; ++i) {
		thrust::copy(testdata.begin() + i * small_hop,
		             testdata.begin() + (i + 1) * small_hop, io_obj.host_in);
		p_rt.process_next_hop_gpu(io_obj.device_in, io_obj.device_out);
		for (std::size_t j = 0; j < small_hop; ++j) {
			EXPECT_NE(io_obj.host_out[j], io_obj.host_in[j]);
		}
	}
}

TEST_F(HPRIOfflineGPUTest, WithPadding)
{
	auto oldsize = testdata.size();

	testdata.resize(testdata.size() + 11);

	auto ret = hpri_offline.process(testdata);
	EXPECT_EQ(ret.size(), testdata.size());
	EXPECT_NE(ret.size(), oldsize);

	for (std::size_t i = 0; i < big_hop * n_big_hops; ++i) {
		EXPECT_NE(ret[i], testdata[i]);
	}
	for (std::size_t i = 0; i < n_small_hops; ++i) {
		thrust::copy(testdata.begin() + i * small_hop,
		             testdata.begin() + (i + 1) * small_hop, io_obj.host_in);
		p_rt.process_next_hop_gpu(io_obj.device_in, io_obj.device_out);
		for (std::size_t j = 0; j < small_hop; ++j) {
			EXPECT_NE(io_obj.host_out[j], io_obj.host_in[j]);
		}
	}
}