#ifndef ZG_CLI_FAKERT
#define ZG_CLI_FAKERT

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>

#include <libzengarden/hps.h>
#include <libzengarden/io.h>

#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>

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

namespace zg {
namespace fakert {

	struct FakeRtParams {
		std::string infile = "";
		std::string outfile = "";
		bool do_hps = false;
		bool cpu = false;
		bool nocopybord = false;
		std::size_t hop = 256;
		float beta = 2.5;
	};

	class FakeRtCommand {
	public:
		FakeRtCommand(FakeRtParams p)
		    : p(p){};

		int validate_params()
		{
			std::cout << "Running zengarden-fakert with the following params:"
			          << "\n\tinfile: " << p.infile
			          << "\n\toutfile: " << p.outfile;

			if (p.do_hps) {
				std::cout << "\n\tdo hps: yes"
				          << "\n\t\thop: " << p.hop << "\n\t\tbeta: " << p.beta;
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
				std::cerr << "fakert params error" << std::endl;
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

			std::vector<float> percussive_out = audio;

			auto chunk_size = p.hop;
			auto chunk_limits = get_chunk_limits(audio, chunk_size);

			std::cout << "Slicing buffer size " << audio.size() << " into "
			          << chunk_limits.size() << " chunks of size "
			          << chunk_size << std::endl;

			std::size_t n = 0;
			float delta_t = 1000 * ( float )p.hop / file_data->sampleRate;

			if (p.cpu) {
				auto hpss = zg::hps::PRealtime<zg::Backend::CPU>(
				    file_data->sampleRate, p.hop, p.beta);

				// not passing an IO object to warmup in the CPU case
				hpss.warmup_cpu();

				float iters = 0.0F;
				int time_tot = 0;

				for (std::vector<std::pair<std::size_t, std::size_t>>::const_iterator
				         chunk_it
				     = chunk_limits.begin();
				     chunk_it != chunk_limits.end(); ++chunk_it) {
					auto t1 = std::chrono::high_resolution_clock::now();

					if (p.do_hps) {
						// process input samples
						hpss.process_next_hop_cpu(
						    audio.data() + chunk_it->first,
						    percussive_out.data() + n);
					}
					else {
						// just loop input back into output
						std::copy(audio.begin() + chunk_it->first,
						          audio.begin() + chunk_it->second,
						          percussive_out.begin() + n);
					}

					auto t2 = std::chrono::high_resolution_clock::now();
					time_tot
					    += std::chrono::duration_cast<std::chrono::microseconds>(
					           t2 - t1)
					           .count();

					n += p.hop;
					iters += 1.0F;
				}

				std::cout << "PRealtime CPU:  Δn = " << p.hop
				          << ", Δt(ms) = " << delta_t
				          << ", average processing duration(us) = "
				          << ( float )time_tot / iters << std::endl;
			}
			else {
				auto hpss = zg::hps::PRealtime<zg::Backend::GPU>(
				    file_data->sampleRate, p.hop, p.beta, p.nocopybord);

				// need an io object to do some warming up
				auto io = zg::io::IOGPU(p.hop);

				hpss.warmup_gpu(io);

				float iters = 0.0F;
				int time_tot = 0;

				for (std::vector<std::pair<std::size_t, std::size_t>>::const_iterator
				         chunk_it
				     = chunk_limits.begin();
				     chunk_it != chunk_limits.end(); ++chunk_it) {
					auto t1 = std::chrono::high_resolution_clock::now();

					if (p.do_hps) {
						// copy input samples into io object
						std::copy(audio.begin() + chunk_it->first,
						          audio.begin() + chunk_it->second, io.host_in);

						// process input samples
						hpss.process_next_hop_gpu(io.device_in, io.device_out);

						// copy output samples from io object
						std::copy(io.host_out, io.host_out + p.hop,
						          percussive_out.begin() + n);
					}
					else {
						// just loop input back into output
						std::copy(audio.begin() + chunk_it->first,
						          audio.begin() + chunk_it->second,
						          percussive_out.begin() + n);
					}

					auto t2 = std::chrono::high_resolution_clock::now();
					time_tot
					    += std::chrono::duration_cast<std::chrono::microseconds>(
					           t2 - t1)
					           .count();

					n += p.hop;
					iters += 1.0F;
				}

				std::cout << "PRealtime GPU:  Δn = " << p.hop
				          << ", Δt(ms) = " << delta_t
				          << ", average processing duration(us) = "
				          << ( float )time_tot / iters << std::endl;
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
		FakeRtParams p;
	};
}; // namespace fakert
}; // namespace zg

#endif /* ZG_CLI_FAKERT */
