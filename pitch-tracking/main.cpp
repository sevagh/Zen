#include <algorithm>
#include <functional>
#include <iostream>

#include "pitch_detection.h"
#include <libnyquist/Decoders.h>

template <class T>
std::vector<T> get_chunks(T container, size_t k)
{
	std::vector<T> ret;

	size_t size = container.size();
	size_t i = 0;

	if (size > k) {
		for (; i < size - k; i += k) {
			ret.push_back(T(container.begin() + i, container.begin() + i + k));
		}
	}

	if (i % k) {
		ret.push_back(T(container.begin() + i, container.begin() + i + (i % k)));
	}

	return ret;
}

int main(int argc, char** argv)
{
	std::size_t chunk_size = 4096;

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

	std::transform(
	    file_data->samples.begin(), file_data->samples.end(),
	    file_data->samples.begin(),
	    std::bind(std::multiplies<float>(), std::placeholders::_1, 10000));

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

	auto sample_rate = file_data->sampleRate;

	auto chunks = get_chunks(audio, chunk_size);

	auto mpm = MPM(chunk_size, sample_rate);

	std::cout << "Slicing buffer size " << audio.size() << " into "
	          << chunks.size() << " chunks of size " << chunk_size << std::endl;

	double t = 0.;
	float timeslice = (( float )chunk_size) / (( float )sample_rate);

	for (auto chunk : chunks) {
		std::cout << "At t: " << t << std::endl;

		auto pitch_mpm = mpm.pitch(chunk);

		std::cout << "\tpitch: " << pitch_mpm << " Hz" << std::endl;

		t += timeslice;
	}

	return 0;
}
