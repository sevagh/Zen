#include <algorithm>
#include <functional>
#include <iostream>
#include <chrono>
#include <thread>
#include <math.h>
#include <cstring>
#include <thread>

#include "rhythm_toolkit/hpss.h"
#include "rhythm_toolkit/io.h"

#include <gflags/gflags.h>
#include <libnyquist/Decoders.h>

#include <soundio/soundio.h>

DEFINE_int32(hop, 256, "hop harmonic (samples)");
DEFINE_double(beta, 2.0, "beta harmonic (separation factor, float)");

// forward declaration - this is at the bottom
static void write_callback(struct SoundIoOutStream* outstream,
                           int frame_count_min,
                           int frame_count_max);

// forward declaration - this is at the bottom
static void underflow_callback(struct SoundIoOutStream* outstream);

std::vector<std::pair<std::size_t, std::size_t>>
get_chunk_limits(std::vector<float> &container, size_t k)
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
		ret.push_back(
		    std::pair<std::size_t, std::size_t>(i, i + (i % k)));
	}

	return ret;
}

class HPSSRtWrapper {
	public:
		std::size_t hop;
		rhythm_toolkit::io::IOGPU io;
		rhythm_toolkit::hpss::PRealtimeGPU hpss;
		std::vector<std::pair<std::size_t, std::size_t>> chunk_limits;
		float iters;
		int time_tot;

		const std::vector<std::pair<std::size_t, std::size_t>>::iterator begin;
		const std::vector<std::pair<std::size_t, std::size_t>>::iterator end;
		std::vector<std::pair<std::size_t, std::size_t>>::iterator curr;

		std::vector<float> &audio;

		HPSSRtWrapper(std::vector<float> &audio, int fs, std::size_t hop, float beta)
			  : hop(hop)
			  , io(rhythm_toolkit::io::IOGPU(hop))
			  , hpss(rhythm_toolkit::hpss::PRealtimeGPU(fs, hop, beta, io))
			  , iters(0.0F)
			  , time_tot(0)
			  , chunk_limits(get_chunk_limits(audio, hop))
			  , begin(chunk_limits.begin())
			  , end(chunk_limits.end())
			  , curr(chunk_limits.begin())
			  , audio(audio)
		{
			std::cout << "Slicing buffer size " << audio.size() << " into "
				  << chunk_limits.size() << " chunks of size " << hop << std::endl;
		}

		int hpss_next_frame() {
			if (curr == end)
				return -1;

			auto t1 = std::chrono::high_resolution_clock::now();

			// copy input samples into io object
			std::copy(audio.begin() + curr->first, audio.begin() + curr->second, io.host_in);

			// process input samples
			hpss.process_next_hop();

			auto t2 = std::chrono::high_resolution_clock::now();
			time_tot += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

			// normalize float samples between -1.0 and 1.0
			auto percussive_limits = std::minmax_element(io.host_out, io.host_out + FLAGS_hop);
			float real_perc_max = std::max(-1*(*percussive_limits.first), *percussive_limits.second);

			for (std::size_t i = 0; i < FLAGS_hop; ++i) {
				io.host_out[i] /= real_perc_max;
			}

			iters += 1.0F;
			curr++;

			return 0;
		}
};


class SoundIoWrapper {
	public:
		int err;
		std::size_t write_size;
		struct SoundIo *soundio;
		struct SoundIoDevice *device;
		struct SoundIoOutStream *outstream;
		struct SoundIoRingBuffer *ringbuf;
		HPSSRtWrapper hpss;

		SoundIoWrapper(std::vector<float> &audio, int fs, std::size_t hop, float beta)
		: hpss(HPSSRtWrapper(audio, fs, hop, beta))
		, write_size(hop * sizeof(float))
		{
			soundio = soundio_create();
			if (!soundio) {
				std::cerr << "out of memory" << std::endl;
				std::exit(-1);
			}

			if ((err = soundio_connect(soundio))) {
				std::cerr << "error connecting: " << soundio_strerror(err) << std::endl;
				std::exit(-1);
			}

			soundio_flush_events(soundio);

			int default_out_device_index = soundio_default_output_device_index(soundio);
			if (default_out_device_index < 0) {
				std::cerr << "no output device found" << std::endl;
				std::exit(-1);
			}

			device = soundio_get_output_device(soundio, default_out_device_index);
			if (!device) {
				std::cerr << "out of memory" << std::endl;
				std::exit(-1);
			}

			outstream = soundio_outstream_create(device);

			outstream->format = SoundIoFormatFloat32NE;
			outstream->write_callback = write_callback;
			outstream->sample_rate = fs;
			outstream->layout = *soundio_channel_layout_get_builtin(SoundIoChannelLayoutIdMono);
			outstream->underflow_callback = underflow_callback;

			if ((err = soundio_outstream_open(outstream))) {
				std::cerr << "unable to open device: " << soundio_strerror(err) << std::endl;
				std::exit(-1);
			}

			if (outstream->layout_error) {
				std::cerr << "unable to set channel layout: " << soundio_strerror(outstream->layout_error) << std::endl;
				std::exit(-1);
			}

			// make a big ringbuffer
			int ringbuf_capacity = 8*65536;
			ringbuf = soundio_ring_buffer_create(soundio, ringbuf_capacity);

			if (!ringbuf) {
				std::cerr << "unable to create ring buffer: out of memory" << std::endl;
				std::exit(-1);
			}

			outstream->userdata = reinterpret_cast<void*>(this);

			char* buf = soundio_ring_buffer_write_ptr(ringbuf);
			std::memset(buf, 0, ringbuf_capacity);
			soundio_ring_buffer_advance_write_ptr(ringbuf, ringbuf_capacity);


		}

		~SoundIoWrapper() {
			soundio_outstream_destroy(outstream);
			soundio_ring_buffer_destroy(ringbuf);
			soundio_device_unref(device);
			soundio_destroy(soundio);
		}

		void start() {
			if ((err = soundio_outstream_start(outstream))) {
				std::cerr << "unable to start output device: " << soundio_strerror(err) << std::endl;
				std::exit(-1);
			}
		}

		void populate_ringbuf_from_hpss() {
			// perform the hpss
			int ret = hpss.hpss_next_frame();
			if (ret == 0) {
				char* buf = soundio_ring_buffer_write_ptr(ringbuf);
				std::size_t fill_count = outstream->software_latency * outstream->sample_rate
						    * outstream->bytes_per_frame;
				fill_count = std::min(fill_count, write_size);

				// copy output samples from io object into soundio output ringbuf
				std::memcpy(buf, hpss.io.host_out, fill_count);
				soundio_ring_buffer_advance_write_ptr(ringbuf, fill_count);
			} else {
				soundio_wakeup(soundio);
			}
		}
};

int
main(int argc, char **argv)
{
	gflags::SetUsageMessage("help\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc != 2) {
		std::cerr << "Usage: wav_analyzer /path/to/audio/file" << std::endl;
		return -1;
	}

	std::cout << "Processing wav file in hops of " << FLAGS_hop
	          << " samples..." << std::endl;

	nqr::NyquistIO loader;

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

	float dt = (float)FLAGS_hop/file_data->sampleRate;

	std::cout << "Processing input signal of size " << audio.size() << " with PRt separation using block " << FLAGS_hop << std::endl;

	auto soundio = SoundIoWrapper(audio, file_data->sampleRate, FLAGS_hop, FLAGS_beta);
	soundio.start();

	// wait for soundio to finish
	for (;;) {
		soundio_wait_events(soundio.soundio);
		break;
	}
	
	std::cout << "PRealtime: Δn = " << FLAGS_hop << ", Δt = " << 1000.0*dt  << " ms, average hpss duration = " << (float)soundio.hpss.time_tot/soundio.hpss.iters/1000.0F << " ms" << std::endl;

	return 0;
}
static void underflow_callback(struct SoundIoOutStream* outstream) {
	SoundIoWrapper *r = reinterpret_cast<SoundIoWrapper*>(outstream->userdata);
	r->populate_ringbuf_from_hpss();
}

static void write_callback(struct SoundIoOutStream* outstream,
                           int frame_count_min,
                           int frame_count_max)
{
	struct SoundIoChannelArea* areas;
	int frames_left;
	int frame_count;
	int err;

	SoundIoWrapper *r = reinterpret_cast<SoundIoWrapper*>(outstream->userdata);
	struct SoundIoRingBuffer* ring_buffer = r->ringbuf;

	char* read_ptr = soundio_ring_buffer_read_ptr(ring_buffer);
	int fill_bytes = soundio_ring_buffer_fill_count(ring_buffer);
	int fill_count = fill_bytes / outstream->bytes_per_frame;

	if (frame_count_min > fill_count) {
		// Ring buffer does not have enough data, fill with zeroes.
		frames_left = frame_count_min;
		for (;;) {
			frame_count = frames_left;
			if (frame_count <= 0)
				return;
			if ((err = soundio_outstream_begin_write(
			         outstream, &areas, &frame_count))) {
				std::cerr << "begin write error: " << soundio_strerror(err) << std::endl;
				std::exit(-1);
			}
				                            
			if (frame_count <= 0)
				return;
			for (int frame = 0; frame < frame_count; frame += 1) {
				for (int ch = 0; ch < outstream->layout.channel_count; ch += 1) {
					std::memset(areas[ch].ptr, 0, outstream->bytes_per_sample);
					areas[ch].ptr += areas[ch].step;
				}
			}
			if ((err = soundio_outstream_end_write(outstream))) {
				std::cerr << "end write error: " << soundio_strerror(err) << std::endl;
				std::exit(-1);
			}
			frames_left -= frame_count;
		}
	}

	int read_count = std::min(frame_count_max, fill_count);
	frames_left = read_count;

	while (frames_left > 0) {
		int frame_count = frames_left;

		if ((err
		     = soundio_outstream_begin_write(outstream, &areas, &frame_count))) {
			std::cerr << "begin write error: " << soundio_strerror(err) << std::endl;
			std::exit(-1);
		}

		if (frame_count <= 0)
			break;

		std::cout << "DRAIN" << std::endl;
		for (int frame = 0; frame < frame_count; frame += 1) {
			for (int ch = 0; ch < outstream->layout.channel_count; ch += 1) {
				std::memcpy(
				    areas[ch].ptr, read_ptr, outstream->bytes_per_sample);
				areas[ch].ptr += areas[ch].step;
				read_ptr += outstream->bytes_per_sample;
			}
		}

		if ((err = soundio_outstream_end_write(outstream))) {
			std::cerr << "end write error: " << soundio_strerror(err) << std::endl;
			std::exit(-1);
		}

		frames_left -= frame_count;
	}

	soundio_ring_buffer_advance_read_ptr(
	    ring_buffer, read_count * outstream->bytes_per_frame);
}
