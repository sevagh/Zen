#include <algorithm>
#include <functional>
#include <iostream>

#include "rhythm_toolkit/hpss.h"
#include <gflags/gflags.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>

DEFINE_int32(hop, 32, "hop (samples)");
DEFINE_double(beta, 2.0, "beta (separation factor, float)");

std::vector<std::vector<float>>
get_chunks(std::vector<float> &container, size_t k)
{
	std::vector<std::vector<float>> ret;

	size_t size = container.size();
	size_t i = 0;

	if (size > k) {
		for (; i < size - k; i += k) {
			ret.push_back(std::vector<float>(container.begin() + i, container.begin() + i + k));
		}
	}

	if (i % k) {
		ret.push_back(
		    std::vector<float>(container.begin() + i, container.begin() + i + (i % k)));
	}

	return ret;
}

int
main(int argc, char **argv)
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

	std::shared_ptr<nqr::AudioData> file_data =
	    std::make_shared<nqr::AudioData>();
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
	} else {
		audio = std::vector<float>(
		    file_data->samples.begin(), file_data->samples.end());
	}

	auto percussive_out = std::vector<float>(audio.size());

	auto chunk_size = FLAGS_hop;
	auto chunks = get_chunks(audio, chunk_size);

	std::cout << "Slicing buffer size " << audio.size() << " into "
	          << chunks.size() << " chunks of size " << chunk_size << std::endl;

	std::size_t n = 0.;

	auto hpss = rhythm_toolkit::hpss::HPSS(file_data->sampleRate, FLAGS_hop, FLAGS_beta);

	for (auto chunk : chunks) {
		std::cout << "At n: " << n << std::endl;
		hpss.process_next_hop(chunk);

		auto perc = hpss.peek_separated_percussive();

		std::copy(perc.begin(), perc.end(), percussive_out.begin() + n);
		n += FLAGS_hop;
	}

	const auto [percussive_min, percussive_max] = std::minmax_element(std::begin(percussive_out), std::end(percussive_out));

	float real_perc_max = std::max(-1*(*percussive_min), *percussive_max);

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
