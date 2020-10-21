#include <gtest/gtest.h>
#include <hps/hps.h>
#include <iostream>
#include <libzen/io.h>
#include <random>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>
#include <thrust/device_vector.h>

#include <cmath>

using namespace zen::internal::hps;
using namespace zen::internal::hps::mfilt;
//using namespace zen::io;
using namespace zen;

// things to test
// copybord, different output types, etc.
// functional test? save for public...

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

static std::vector<float> generate_data_huge(size_t size)
{
	// use realistic values of normalized input floats
	static std::uniform_real_distribution<float> distribution(
	    std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
	static std::default_random_engine generator;

	std::vector<float> data(size);

	std::generate(
	    data.begin(), data.end(), []() { return distribution(generator); });

	return data;
}

zen::io::IOGPU io_obj(8192);

class HPRTest : public ::testing::Test {

public:
	std::size_t hop;
	std::size_t n_hops;

	// test every variant
	HPR<Backend::GPU> hpr_causal_g;
	HPR<Backend::GPU> hpr_causal_g_nocopybord;
	HPR<Backend::CPU> hpr_causal_c;
	HPR<Backend::CPU> hpr_causal_c_nocopybord;

	HPR<Backend::GPU> hpr_anticausal_g;
	HPR<Backend::GPU> hpr_anticausal_g_nocopybord;
	HPR<Backend::CPU> hpr_anticausal_c;
	HPR<Backend::CPU> hpr_anticausal_c_nocopybord;

	std::vector<float> testdata_small;
	std::vector<float> testdata_huge;

	HPRTest(std::size_t hop,
	        std::size_t n_hops,
	        bool copybord,
	        unsigned int out_flags)
	    : hop(hop)
	    , n_hops(n_hops)
	    , hpr_causal_g(48000.0F,
	                   hop,
	                   2.0,
	                   out_flags,
	                   MedianFilterDirection::TimeCausal,
	                   true)
	    , hpr_causal_g_nocopybord(48000.0F,
	                              hop,
	                              2.0,
	                              out_flags,
	                              MedianFilterDirection::TimeCausal,
	                              false)
	    , hpr_causal_c(48000.0F,
	                   hop,
	                   2.0,
	                   out_flags,
	                   MedianFilterDirection::TimeCausal,
	                   true)
	    , hpr_causal_c_nocopybord(48000.0F,
	                              hop,
	                              2.0,
	                              out_flags,
	                              MedianFilterDirection::TimeCausal,
	                              false)
	    , hpr_anticausal_g(48000.0F,
	                       hop,
	                       2.0,
	                       out_flags,
	                       MedianFilterDirection::TimeAnticausal,
	                       true)
	    , hpr_anticausal_g_nocopybord(48000.0F,
	                                  hop,
	                                  2.0,
	                                  out_flags,
	                                  MedianFilterDirection::TimeAnticausal,
	                                  false)
	    , hpr_anticausal_c(48000.0F,
	                       hop,
	                       2.0,
	                       out_flags,
	                       MedianFilterDirection::TimeAnticausal,
	                       true)
	    , hpr_anticausal_c_nocopybord(48000.0F,
	                                  hop,
	                                  2.0,
	                                  out_flags,
	                                  MedianFilterDirection::TimeAnticausal,
	                                  false)
	    , testdata_small(generate_data_normalized(n_hops * hop))
	    , testdata_huge(generate_data_huge(n_hops * hop)){};

	virtual void SetUp() {}

	virtual void TearDown() {}
};

// typical hop is 256, from the driedger and fitzenerald papers
// create an audio clip that's ~25.6k samples long, about 0.25s
class HPRAllOutputTest : public HPRTest {
protected:
	HPRAllOutputTest()
	    : HPRTest(256, 100, true, HPSS_HARMONIC | HPSS_PERCUSSIVE | HPSS_RESIDUAL)
	{
	}
};

// tests with only percussive output (which is the only other common variant of HPS)
class HPRPercTest : public HPRTest {
protected:
	HPRPercTest()
	    : HPRTest(256, 100, true, HPSS_PERCUSSIVE)
	{
	}
};

TEST_F(HPRAllOutputTest, ProcessingModifiesInput)
{
	for (std::size_t i = 0; i < n_hops; ++i) {
		thrust::copy(testdata_small.begin() + i * hop,
		             testdata_small.begin() + (i + 1) * hop, io_obj.host_in);

		// TEST ALL VARIANTS
		hpr_causal_g.process_next_hop(io_obj.device_in);
		hpr_causal_g_nocopybord.process_next_hop(io_obj.device_in);
		hpr_causal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_causal_c_nocopybord.process_next_hop(testdata_small.data()
		                                         + i * hop);
		hpr_anticausal_g.process_next_hop(io_obj.device_in);
		hpr_anticausal_g_nocopybord.process_next_hop(io_obj.device_in);
		hpr_anticausal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_anticausal_c_nocopybord.process_next_hop(testdata_small.data()
		                                             + i * hop);

		for (std::size_t j = 0; j < hop; ++j) {
			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_causal_c.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_causal_c_nocopybord.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c_nocopybord.percussive_out[j]);

			EXPECT_NE(io_obj.device_in[j], hpr_causal_g.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_causal_g_nocopybord.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j], hpr_anticausal_g.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_anticausal_g_nocopybord.percussive_out[j]);

			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_causal_c.harmonic_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_causal_c_nocopybord.harmonic_out[j]);
			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_anticausal_c.harmonic_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c_nocopybord.harmonic_out[j]);

			EXPECT_NE(io_obj.device_in[j], hpr_causal_g.harmonic_out[j]);
			EXPECT_NE(
			    io_obj.device_in[j], hpr_causal_g_nocopybord.harmonic_out[j]);
			EXPECT_NE(io_obj.device_in[j], hpr_anticausal_g.harmonic_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_anticausal_g_nocopybord.harmonic_out[j]);

			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_causal_c.residual_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_causal_c_nocopybord.residual_out[j]);
			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_anticausal_c.residual_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c_nocopybord.residual_out[j]);

			EXPECT_NE(io_obj.device_in[j], hpr_causal_g.residual_out[j]);
			EXPECT_NE(
			    io_obj.device_in[j], hpr_causal_g_nocopybord.residual_out[j]);
			EXPECT_NE(io_obj.device_in[j], hpr_anticausal_g.residual_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_anticausal_g_nocopybord.residual_out[j]);
		}
	}
}

TEST_F(HPRAllOutputTest, AnticausalVsCausalCopybordVsNocopybord)
{
	for (std::size_t i = 0; i < n_hops; ++i) {
		thrust::copy(testdata_small.begin() + i * hop,
		             testdata_small.begin() + (i + 1) * hop, io_obj.host_in);

		/* TEST COPYBORD VS NOCOPYBORD */

		// for GPUs, it should change everything
		hpr_causal_g.process_next_hop(io_obj.device_in);
		hpr_causal_g_nocopybord.process_next_hop(io_obj.device_in);

		// for CPUs, they're both the same
		hpr_causal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_causal_c_nocopybord.process_next_hop(testdata_small.data()
		                                         + i * hop);

		/* TEST CAUSAL VS ANTICAUSAL */

		// different on both CPU and GPU
		hpr_anticausal_g.process_next_hop(io_obj.device_in);
		hpr_anticausal_g_nocopybord.process_next_hop(io_obj.device_in);
		hpr_anticausal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_anticausal_c_nocopybord.process_next_hop(testdata_small.data()
		                                             + i * hop);

		// then verify copybord behavior on cpu/gpu
		for (std::size_t j = 0; j < hop; ++j) {
			EXPECT_NE(hpr_causal_g.percussive_out[j],
			          hpr_causal_g_nocopybord.percussive_out[j]);
			EXPECT_EQ(hpr_causal_c.percussive_out[j],
			          hpr_causal_c_nocopybord.percussive_out[j]);
		}

		// then verify anti-causal/causal + copybord/nocopybord behavior on cpu/gpu
		for (std::size_t j = 0; j < hop; ++j) {
			// if the causal output is 0, so is the anticausal
			if (hpr_causal_g.percussive_out[j] == 0) {
				EXPECT_EQ(hpr_causal_g.percussive_out[j],
				          hpr_anticausal_g.percussive_out[j]);
			}
			else {
				EXPECT_NE(hpr_causal_g.percussive_out[j],
				          hpr_anticausal_g.percussive_out[j]);
			}

			EXPECT_NE(hpr_causal_c.percussive_out[j],
			          hpr_anticausal_c.percussive_out[j]);

			/* TEST ANTICAUSAL COPYBORD VS NOCOPYBORD CPU */
			EXPECT_EQ(hpr_anticausal_c.percussive_out[j],
			          hpr_anticausal_c_nocopybord.percussive_out[j]);
		}
	}
}

TEST_F(HPRPercTest, PercOnlyOutputsPerc)
{
	for (std::size_t i = 0; i < n_hops; ++i) {
		thrust::copy(testdata_small.begin() + i * hop,
		             testdata_small.begin() + (i + 1) * hop, io_obj.host_in);

		// TEST ALL VARIANTS
		hpr_causal_g.process_next_hop(io_obj.device_in);
		hpr_causal_g_nocopybord.process_next_hop(io_obj.device_in);
		hpr_causal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_causal_c_nocopybord.process_next_hop(testdata_small.data()
		                                         + i * hop);
		hpr_anticausal_g.process_next_hop(io_obj.device_in);
		hpr_anticausal_g_nocopybord.process_next_hop(io_obj.device_in);
		hpr_anticausal_c.process_next_hop(testdata_small.data() + i * hop);
		hpr_anticausal_c_nocopybord.process_next_hop(testdata_small.data()
		                                             + i * hop);

		for (std::size_t j = 0; j < hop; ++j) {
			EXPECT_NE(
			    testdata_small[i * hop + j], hpr_causal_c.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_causal_c_nocopybord.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c.percussive_out[j]);
			EXPECT_NE(testdata_small[i * hop + j],
			          hpr_anticausal_c_nocopybord.percussive_out[j]);

			EXPECT_NE(io_obj.device_in[j], hpr_causal_g.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_causal_g_nocopybord.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j], hpr_anticausal_g.percussive_out[j]);
			EXPECT_NE(io_obj.device_in[j],
			          hpr_anticausal_g_nocopybord.percussive_out[j]);

			EXPECT_EQ(hpr_causal_c.harmonic_out[j], 0);
			EXPECT_EQ(hpr_causal_c.residual_out[j], 0);

			EXPECT_EQ(hpr_causal_c_nocopybord.harmonic_out[j], 0);
			EXPECT_EQ(hpr_causal_c_nocopybord.residual_out[j], 0);

			EXPECT_EQ(hpr_causal_g.harmonic_out[j], 0);
			EXPECT_EQ(hpr_causal_g.residual_out[j], 0);

			EXPECT_EQ(hpr_causal_g_nocopybord.harmonic_out[j], 0);
			EXPECT_EQ(hpr_causal_g_nocopybord.residual_out[j], 0);

			EXPECT_EQ(hpr_anticausal_g.harmonic_out[j], 0);
			EXPECT_EQ(hpr_anticausal_g.residual_out[j], 0);

			EXPECT_EQ(hpr_anticausal_g_nocopybord.harmonic_out[j], 0);
			EXPECT_EQ(hpr_anticausal_g_nocopybord.residual_out[j], 0);

			EXPECT_EQ(hpr_anticausal_c.harmonic_out[j], 0);
			EXPECT_EQ(hpr_anticausal_c.residual_out[j], 0);

			EXPECT_EQ(hpr_anticausal_c_nocopybord.harmonic_out[j], 0);
			EXPECT_EQ(hpr_anticausal_c_nocopybord.residual_out[j], 0);
		}
	}
}

TEST_F(HPRPercTest, ResettingDoesTheRightThing)
{
	std::size_t i = 0;

	std::vector<float> im(hop);

	thrust::copy(testdata_small.begin() + i * hop,
	             testdata_small.begin() + (i + 1) * hop, io_obj.host_in);

	// test reset buffers behavior on just one variant
	hpr_causal_g.process_next_hop(io_obj.device_in);

	thrust::copy(hpr_causal_g.percussive_out.begin(),
	             hpr_causal_g.percussive_out.begin() + hop, io_obj.device_out);

	std::copy(io_obj.host_out, io_obj.host_out + hop, im.begin());
	hpr_causal_g.reset_buffers();

	hpr_causal_g.process_next_hop(io_obj.device_in);

	// reset, run again, expect identical output
	for (std::size_t j = 0; j < hop; ++j) {
		EXPECT_EQ(hpr_causal_g.percussive_out[j], im[j]);
	}
}
