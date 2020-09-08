#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>

#include "rhythm_toolkit/hpss.h"
#include "rhythm_toolkit/io.h"

#include <gflags/gflags.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>

DEFINE_int32(hop, 256, "hop (samples)");
DEFINE_double(beta, 2.6, "beta (separation factor, float)");

std::vector<std::pair<std::size_t, std::size_t>>
get_chunk_limits(std::vector<float>& container, size_t k)
{
	std::vector<std::pair<std::size_t, std::size_t>> ret;

	size_t size = container.size();
	size_t i = 0;

	if (size > k) {
		for (; i < size - k; i += k) {
			ret.push_back(std::pair<std::size_t, std::size_t>{i, i + k});
		}
	}

	if (i % k) {
		ret.push_back(std::pair<std::size_t, std::size_t>(i, i + (i % k)));
	}

	return ret;
}

int main(int argc, char** argv)
{
	gflags::SetUsageMessage("help\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::cout << "Processing wav file in hops of " << FLAGS_hop
	          << " samples..." << std::endl;

	nqr::NyquistIO loader;

	if (argc != 2) {
		std::cerr << "Usage: wav_analyzer /path/to/audio/file" << std::endl;
		return -1;
	}

	std::shared_ptr<nqr::AudioData> file_data
	    = std::make_shared<nqr::AudioData>();
	loader.Load(file_data.get(), argv[1]);

	std::cout << "Audio file info:" << std::endl;

	std::cout << "\tsample rate: " << file_data->sampleRate << std::endl;
	std::cout << "\tlen samples: " << file_data->samples.size() << std::endl;
	std::cout << "\tframe size: " << file_data->frameSize << std::endl;
	std::cout << "\tseconds: " << file_data->lengthSeconds << std::endl;
	std::cout << "\tchannels: " << file_data->channelCount << std::endl;

	std::vector<float> audio;

	if (file_data->channelCount == 2) {
		// convert stereo to mono
		std::vector<float> audio_copy(file_data->samples.size() / 2);
		nqr::StereoToMono(file_data->samples.data(), audio_copy.data(),
		                  file_data->samples.size());
		audio = std::vector<float>(audio_copy.begin(), audio_copy.end());
	}
	else {
		audio = std::vector<float>(
		    file_data->samples.begin(), file_data->samples.end());
	}

	auto percussive_out = std::vector<float>(audio.size());

	auto chunk_size = FLAGS_hop;
	auto chunk_limits = get_chunk_limits(audio, chunk_size);

	std::cout << "Slicing buffer size " << audio.size() << " into "
	          << chunk_limits.size() << " chunks of size " << chunk_size
	          << std::endl;

	std::size_t n = 0;
	float delta_t = 1000 * ( float )FLAGS_hop / file_data->sampleRate;

	auto io = rhythm_toolkit::io::IOGPU(FLAGS_hop);
	auto hpss = rhythm_toolkit::hpss::PRealtimeGPU(
	    file_data->sampleRate, FLAGS_hop, FLAGS_beta, io);

	hpss.warmup();

	float iters = 0.0F;
	int time_tot = 0;

	for (std::vector<std::pair<std::size_t, std::size_t>>::const_iterator chunk_it
	     = chunk_limits.begin();
	     chunk_it != chunk_limits.end(); ++chunk_it) {
		auto t1 = std::chrono::high_resolution_clock::now();

		// copy input samples into io object
		std::copy(audio.begin() + chunk_it->first,
		          audio.begin() + chunk_it->second, io.host_in);

		// process input samples
		hpss.process_next_hop();

		// copy output samples from io object
		std::copy(
		    io.host_out, io.host_out + FLAGS_hop, percussive_out.begin() + n);

		auto t2 = std::chrono::high_resolution_clock::now();
		time_tot
		    += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
		           .count();

		n += FLAGS_hop;
		iters += 1.0F;
	}

	std::cout << "Δn = " << FLAGS_hop << ", Δt(ms) = " << delta_t
	          << ", average hpss duration(us) = " << ( float )time_tot / iters
	          << std::endl;

	auto percussive_limits = std::minmax_element(
	    std::begin(percussive_out), std::end(percussive_out));
	float real_perc_max
	    = std::max(-1 * (*percussive_limits.first), *percussive_limits.second);

	// normalize between -1.0 and 1.0
	for (std::size_t i = 0; i < audio.size(); ++i) {
		percussive_out[i] /= real_perc_max;
	}

	nqr::EncoderParams encoder_params{
	    1,
	    nqr::PCMFormat::PCM_16,
	    nqr::DitherType::DITHER_NONE,
	};

	const nqr::AudioData perc_out{
	    1,
	    file_data->sampleRate,
	    file_data->lengthSeconds,
	    file_data->frameSize,
	    percussive_out,
	    file_data->sourceFormat,
	};

	nqr::encode_wav_to_disk(encoder_params, &perc_out, "./perc_out.wav");

	return 0;
}
