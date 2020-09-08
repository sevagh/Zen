#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <math.h>

#include <libzengarden/hps.h>
#include <libzengarden/io.h>

#include <gflags/gflags.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>

DEFINE_int32(hop_h, 4096, "hop harmonic (samples)");
DEFINE_int32(hop_p, 256, "hop percussive (samples)");
DEFINE_double(beta_h, 2.0, "beta harmonic (separation factor, float)");
DEFINE_double(beta_p, 2.0, "beta harmonic (separation factor, float)");
DEFINE_bool(cpu, false, "use CPU variant instead of GPU by default");

int main(int argc, char** argv)
{
	gflags::SetUsageMessage("help\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc != 2) {
		std::cerr << "Usage: wav_analyzer /path/to/audio/file" << std::endl;
		return -1;
	}

	std::cout << "Processing wav file in hops of " << FLAGS_hop_h
	          << " samples..." << std::endl;

	nqr::NyquistIO loader;

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
	std::cout << "Processing input signal of size " << audio.size()
	          << " with HPR-I separation using blocks of " << FLAGS_hop_h
	          << ", " << FLAGS_hop_p << std::endl;

	std::vector<float> percussive_out;

	if (!FLAGS_cpu) {
		auto hpss
		    = zg::hps::HPRIOfflineGPU(file_data->sampleRate, FLAGS_hop_h,
		                              FLAGS_hop_p, FLAGS_beta_h, FLAGS_beta_p);

		auto t1 = std::chrono::high_resolution_clock::now();
		percussive_out = hpss.process(audio);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto dur
		    = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
		          .count();

		std::cout << "GPU/CUDA/thrust: 2-pass HPR-I-Offline took " << dur
		          << " ms" << std::endl;
	}
	else {
		auto hpss
		    = zg::hps::HPRIOfflineCPU(file_data->sampleRate, FLAGS_hop_h,
		                              FLAGS_hop_p, FLAGS_beta_h, FLAGS_beta_p);

		auto t1 = std::chrono::high_resolution_clock::now();
		percussive_out = hpss.process(audio);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto dur
		    = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
		          .count();

		std::cout << "CPU/IPP: 2-pass HPR-I-Offline took " << dur << " ms"
		          << std::endl;
	}

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
