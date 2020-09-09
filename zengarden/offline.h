#ifndef ZG_CLI_OFFLINE
#define ZG_CLI_OFFLINE

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <math.h>

#include <libzengarden/hps.h>
#include <libzengarden/io.h>

#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>

namespace zg {
namespace offline {
	struct OfflineParams {
		std::string infile = "";
		std::string outfile = "";
		bool do_hps = false;
		bool cpu = false;
		std::size_t hop_h = 4096;
		std::size_t hop_p = 256;
		float beta_h = 2.0;
		float beta_p = 2.0;
	};

	class OfflineCommand {
	public:
		OfflineCommand(OfflineParams p)
		    : p(p){};

		int validate_params()
		{
			std::cout << "Running zengarden-offline with the following params:"
			          << "\n\tinfile: " << p.infile
			          << "\n\toutfile: " << p.outfile;

			if (p.do_hps) {
				std::cout << "\n\tdo hps: yes"
				          << "\n\t\tharmonic hop: " << p.hop_h
				          << "\n\t\tharmonic beta: " << p.beta_h
				          << "\n\t\tpercussive hop: " << p.hop_p
				          << "\n\t\tpercussive beta: " << p.beta_p;
			}
			else {
				std::cout << "\n\tdo hps: no";
			}

			if (p.cpu) {
				std::cout << "\n\tcompute: cpu (ipp)";
			}
			else {
				std::cout << "\n\tcompute: gpu (cuda/npp)";
			}

			std::cout << std::endl;

			return 0;
		}

		int execute()
		{
			if (validate_params() != 0) {
				std::cerr << "offline params error" << std::endl;
				std::exit(1);
			}

			nqr::NyquistIO loader;

			std::shared_ptr<nqr::AudioData> file_data
			    = std::make_shared<nqr::AudioData>();
			loader.Load(file_data.get(), p.infile);

			std::cout << "Audio file info:" << std::endl;

			std::cout << "\tsample rate: " << file_data->sampleRate
			          << std::endl;
			std::cout << "\tlen samples: " << file_data->samples.size()
			          << std::endl;
			std::cout << "\tframe size: " << file_data->frameSize << std::endl;
			std::cout << "\tseconds: " << file_data->lengthSeconds << std::endl;
			std::cout << "\tchannels: " << file_data->channelCount << std::endl;

			std::vector<float> audio;

			if (file_data->channelCount == 2) {
				// convert stereo to mono
				std::vector<float> audio_copy(file_data->samples.size() / 2);
				nqr::StereoToMono(file_data->samples.data(), audio_copy.data(),
				                  file_data->samples.size());
				audio
				    = std::vector<float>(audio_copy.begin(), audio_copy.end());
			}
			else {
				audio = std::vector<float>(
				    file_data->samples.begin(), file_data->samples.end());
			}

			std::vector<float> percussive_out;

			if (p.do_hps) {
				std::cout << "Processing input signal of size " << audio.size()
				          << " with HPR-I separation using harmonic params: "
				          << p.hop_h << "," << p.beta_h
				          << ", percussive params: " << p.hop_p << ","
				          << p.beta_p << std::endl;

				if (p.cpu) {
					auto hpss = zg::hps::HPRIOfflineCPU(file_data->sampleRate,
					                                    p.hop_h, p.hop_p,
					                                    p.beta_h, p.beta_p);

					auto t1 = std::chrono::high_resolution_clock::now();
					percussive_out = hpss.process(audio);
					auto t2 = std::chrono::high_resolution_clock::now();
					auto dur
					    = std::chrono::duration_cast<std::chrono::milliseconds>(
					          t2 - t1)
					          .count();

					std::cout << "CPU/IPP: 2-pass HPR-I-Offline took " << dur
					          << " ms" << std::endl;
				}
				else {
					auto hpss = zg::hps::HPRIOfflineGPU(file_data->sampleRate,
					                                    p.hop_h, p.hop_p,
					                                    p.beta_h, p.beta_p);

					auto t1 = std::chrono::high_resolution_clock::now();
					percussive_out = hpss.process(audio);
					auto t2 = std::chrono::high_resolution_clock::now();
					auto dur
					    = std::chrono::duration_cast<std::chrono::milliseconds>(
					          t2 - t1)
					          .count();

					std::cout << "GPU/CUDA/thrust: 2-pass HPR-I-Offline took "
					          << dur << " ms" << std::endl;
				}
			}
			else {
				percussive_out = audio;
			}

			if (p.outfile != "") {
				auto percussive_limits = std::minmax_element(
				    std::begin(percussive_out), std::end(percussive_out));

				float real_perc_max = std::max(-1 * (*percussive_limits.first),
				                               *percussive_limits.second);

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

				nqr::encode_wav_to_disk(encoder_params, &perc_out, p.outfile);
			}

			return 0;
		}

	private:
		OfflineParams p;
	};
}; // namespace offline
}; // namespace zg

#endif /* ZG_CLI_OFFLINE */
