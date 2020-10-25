#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>

#include "BTrack.h"
#include <libnyquist/Decoders.h>

#include <libzen/hps.h>
#include <libzen/io.h>

static std::vector<std::pair<std::size_t, std::size_t>>
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
	std::cout << std::fixed;
	std::cout << std::setprecision(2);

	std::size_t chunk_size = 512;

	std::cout << "Slicing wav file into chunks of " << chunk_size
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
	std::vector<float> percussive_out = audio;

	auto sample_rate = file_data->sampleRate;

	auto chunk_limits = get_chunk_limits(audio, chunk_size);

	// this implementation of BTrack is fixed for a hop of 512
	auto btrack1 = BTrack(sample_rate);
	auto btrack2 = BTrack(sample_rate);

	std::cout << "Slicing buffer size " << audio.size() << " into "
	          << chunk_limits.size() << " chunks of size " << chunk_size << std::endl;

	double t = 0.;
	float timeslice = (( float )chunk_size) / (( float )sample_rate);

	auto hpss = zen::hps::HPRRealtime<zen::Backend::GPU>(
	    file_data->sampleRate, chunk_size, 2.5,
	    zen::hps::OUTPUT_PERCUSSIVE);

	// need an io object to do some warming up
	auto io = zen::io::IOGPU(chunk_size);
	hpss.warmup(io);
	std::size_t n = 0;

	for (std::vector<std::pair<std::size_t, std::size_t>>::const_iterator
				         chunk_it
				     = chunk_limits.begin();
				     chunk_it != chunk_limits.end(); ++chunk_it) {
		// copy input samples into io object
		std::copy(audio.data() + chunk_it->first,
			  audio.data() + chunk_it->second, io.host_in);

		// process input samples
		hpss.process_next_hop(io.device_in);
		hpss.copy_percussive(io.device_out);

		// copy output samples from io object
		std::copy(io.host_out, io.host_out + chunk_size,
			  percussive_out.begin() + n);

		btrack1.processHop(percussive_out.data() + chunk_it->first);

		btrack2.processHop(audio.data() + chunk_it->first);

		std::cout << "t: " << t << ",\t" << "beat? (+HPR): "  << btrack1.beatDueInFrame() << ",\tbeat? (-HPR): " << btrack2.beatDueInFrame << "\n";

		t += timeslice;
		n += chunk_size;
	}

	return 0;
}
