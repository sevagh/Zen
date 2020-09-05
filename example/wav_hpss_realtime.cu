#include <soundio/soundio.h>

#include <chrono>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>

//#include <gflags/gflags.h>
#include "rhythm_toolkit/hpss.h"
#include "rhythm_toolkit/io.h"

__attribute__((cold)) __attribute__((noreturn))
__attribute__((format(printf, 1, 2))) static void
panic(const char* format, ...)
{
	va_list ap;
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	abort();
}

class BufferedHPSS {
public:
	struct SoundIoRingBuffer* ring_buffer_in;
	struct SoundIoRingBuffer* ring_buffer_out;

	BufferedHPSS(float latency_seconds,
	             std::size_t ringbuf_capacity,
	             std::size_t hpss_hop_size,
	             float beta,
	             int fs,
	             struct SoundIo* soundio)
	    : hop(hpss_hop_size)
	    , latency_us(( int )(1000000.0F * latency_seconds))
	    , capacity(ringbuf_capacity)
	    , io(rhythm_toolkit::io::IOGPU(hpss_hop_size))
	    , hpss(rhythm_toolkit::hpss::PRealtimeGPU(( float )fs,
	                                              hpss_hop_size,
	                                              beta,
	                                              io))
	{
		ring_buffer_in = soundio_ring_buffer_create(soundio, capacity);
		if (!ring_buffer_in)
			panic("unable to create ring buffer: out of memory");

		ring_buffer_out = soundio_ring_buffer_create(soundio, capacity);
		if (!ring_buffer_out)
			panic("unable to create ring buffer: out of memory");

		std::cout << "warming up HPSS first..." << std::endl;
		warmup_hpss();
	}

	// 1000 iterations to warm up the gpu apparatus
	void warmup_hpss()
	{
		int test_iters = 1000;
		std::vector<float> testdata(test_iters * hop);
		std::vector<float> outdata(test_iters * hop);
		std::iota(testdata.begin(), testdata.end(), 0.0F);

		for (std::size_t i = 0; i < test_iters; ++i) {
			thrust::copy(testdata.begin() + i * hop,
			             testdata.begin() + (i + 1) * hop, io.host_in);
			hpss.process_next_hop();
			thrust::copy(
			    io.host_out, io.host_out + hop, outdata.begin() + i * hop);
		}
	}

	~BufferedHPSS()
	{
		soundio_ring_buffer_destroy(ring_buffer_in);
		soundio_ring_buffer_destroy(ring_buffer_out);
	}

	void do_hpss()
	{
		for (;;) {
			int input_count = soundio_ring_buffer_fill_count(ring_buffer_in)
			                  / sizeof(float);
			int output_free_count
			    = soundio_ring_buffer_free_count(ring_buffer_out)
			      / sizeof(float);

			// if there are at least hop samples to write to the hpss object
			// and at least hop space in the output ringbuffer in which to
			// write the results
			if (input_count >= hop && output_free_count >= hop) {
				std::size_t rw_amount = hop * sizeof(float);

				char* read_ptr = soundio_ring_buffer_read_ptr(ring_buffer_in);

				memcpy(( char* )io.host_in, read_ptr, rw_amount);
				soundio_ring_buffer_advance_read_ptr(ring_buffer_in, rw_amount);

				hpss.process_next_hop();

				auto percussive_limits
				    = std::minmax_element(io.host_out, io.host_out + hop);

				float real_perc_max = std::max(-1 * (*percussive_limits.first),
				                               *percussive_limits.second);

				// normalize between -1.0 and 1.0
				for (std::size_t i = 0; i < hop; ++i) {
					io.host_out[i] /= real_perc_max;
				}

				char* write_ptr
				    = soundio_ring_buffer_write_ptr(ring_buffer_out);

				memcpy(write_ptr, ( char* )io.host_out, rw_amount);

				soundio_ring_buffer_advance_write_ptr(
				    ring_buffer_out, rw_amount);
				input_count = soundio_ring_buffer_fill_count(ring_buffer_in);
			}
		}
	}

private:
	int latency_us;
	std::size_t hop;
	std::size_t capacity;
	rhythm_toolkit::io::IOGPU io;
	rhythm_toolkit::hpss::PRealtimeGPU hpss;
};

BufferedHPSS* hpss = nullptr;

static int min_int(int a, int b) { return (a < b) ? a : b; }

static void read_callback(struct SoundIoInStream* instream,
                          int frame_count_min,
                          int frame_count_max)
{
	struct SoundIoChannelArea* areas;
	int err;
	char* write_ptr = soundio_ring_buffer_write_ptr(hpss->ring_buffer_in);
	int free_bytes = soundio_ring_buffer_free_count(hpss->ring_buffer_in);
	int free_count = free_bytes / instream->bytes_per_frame;

	if (frame_count_min > free_count)
		panic("ring buffer overflow");

	int write_frames = min_int(free_count, frame_count_max);
	int frames_left = write_frames;

	for (;;) {
		int frame_count = frames_left;

		if ((err = soundio_instream_begin_read(instream, &areas, &frame_count)))
			panic("begin read error: %s", soundio_strerror(err));

		if (!frame_count)
			break;

		if (!areas) {
			// Due to an overflow there is a hole. Fill the ring buffer with
			// silence for the size of the hole.
			memset(write_ptr, 0, frame_count * instream->bytes_per_frame);
			fprintf(stderr, "Dropped %d frames due to internal overflow\n",
			        frame_count);
		}
		else {
			for (int frame = 0; frame < frame_count; frame += 1) {
				memcpy(write_ptr, areas[0].ptr, instream->bytes_per_sample);
				areas[0].ptr += areas[0].step;
				write_ptr += instream->bytes_per_sample;
			}
		}

		if ((err = soundio_instream_end_read(instream)))
			panic("end read error: %s", soundio_strerror(err));

		frames_left -= frame_count;
		if (frames_left <= 0)
			break;
	}

	int advance_bytes = write_frames * instream->bytes_per_frame;
	soundio_ring_buffer_advance_write_ptr(hpss->ring_buffer_in, advance_bytes);
}

static void write_callback(struct SoundIoOutStream* outstream,
                           int frame_count_min,
                           int frame_count_max)
{
	struct SoundIoChannelArea* areas;
	int frames_left;
	int frame_count;
	int err;

	char* read_ptr = soundio_ring_buffer_read_ptr(hpss->ring_buffer_out);
	int fill_bytes = soundio_ring_buffer_fill_count(hpss->ring_buffer_out);
	int fill_count = fill_bytes / outstream->bytes_per_frame;

	if (frame_count_min > fill_count) {
		// Ring buffer does not have enough data, fill with zeroes.
		frames_left = frame_count_min;
		for (;;) {
			frame_count = frames_left;
			if (frame_count <= 0)
				return;
			if ((err = soundio_outstream_begin_write(
			         outstream, &areas, &frame_count)))
				panic("begin write error: %s", soundio_strerror(err));
			if (frame_count <= 0)
				return;
			for (int frame = 0; frame < frame_count; frame += 1) {
				memset(areas[0].ptr, 0, outstream->bytes_per_sample);
				areas[0].ptr += areas[0].step;
			}
			if ((err = soundio_outstream_end_write(outstream)))
				panic("end write error: %s", soundio_strerror(err));
			frames_left -= frame_count;
		}
	}

	int read_count = min_int(frame_count_max, fill_count);
	frames_left = read_count;

	while (frames_left > 0) {
		int frame_count = frames_left;

		if ((err
		     = soundio_outstream_begin_write(outstream, &areas, &frame_count)))
			panic("begin write error: %s", soundio_strerror(err));

		if (frame_count <= 0)
			break;

		for (int frame = 0; frame < frame_count; frame += 1) {
			memcpy(areas[0].ptr, read_ptr, outstream->bytes_per_sample);
			areas[0].ptr += areas[0].step;
			read_ptr += outstream->bytes_per_sample;
		}

		if ((err = soundio_outstream_end_write(outstream)))
			panic("end write error: %s", soundio_strerror(err));

		frames_left -= frame_count;
	}

	soundio_ring_buffer_advance_read_ptr(
	    hpss->ring_buffer_out, read_count * outstream->bytes_per_frame);
}

static void underflow_callback(struct SoundIoOutStream* outstream)
{
	static int count = 0;
	// fprintf(stderr, "underflow %d\n", ++count);
}

static int usage(char* exe)
{
	fprintf(stderr,
	        "Usage: %s [options]\n"
	        "Options:\n"
	        "  [--backend dummy|alsa|pulseaudio|jack|coreaudio|wasapi]\n"
	        "  [--in-device id]\n"
	        "  [--in-raw]\n"
	        "  [--out-device id]\n"
	        "  [--out-raw]\n"
	        "  [--latency samples]\n",
	        exe);
	return 1;
}

int main(int argc, char** argv)
{
	char* exe = argv[0];
	enum SoundIoBackend backend = SoundIoBackendNone;
	char* in_device_id = NULL;
	char* out_device_id = NULL;
	bool in_raw = false;
	bool out_raw = false;

	double microphone_latency = 0.02; // seconds
	std::size_t hpss_hop = 1024;      // samples
	float beta = 2.0;

	for (int i = 1; i < argc; i += 1) {
		char* arg = argv[i];
		if (arg[0] == '-' && arg[1] == '-') {
			if (strcmp(arg, "--in-raw") == 0) {
				in_raw = true;
			}
			else if (strcmp(arg, "--out-raw") == 0) {
				out_raw = true;
			}
			else if (++i >= argc) {
				return usage(exe);
			}
			else if (strcmp(arg, "--backend") == 0) {
				if (strcmp("dummy", argv[i]) == 0) {
					backend = SoundIoBackendDummy;
				}
				else if (strcmp("alsa", argv[i]) == 0) {
					backend = SoundIoBackendAlsa;
				}
				else if (strcmp("pulseaudio", argv[i]) == 0) {
					backend = SoundIoBackendPulseAudio;
				}
				else if (strcmp("jack", argv[i]) == 0) {
					backend = SoundIoBackendJack;
				}
				else if (strcmp("coreaudio", argv[i]) == 0) {
					backend = SoundIoBackendCoreAudio;
				}
				else if (strcmp("wasapi", argv[i]) == 0) {
					backend = SoundIoBackendWasapi;
				}
				else {
					fprintf(stderr, "Invalid backend: %s\n", argv[i]);
					return 1;
				}
			}
			else if (strcmp(arg, "--in-device") == 0) {
				in_device_id = argv[i];
			}
			else if (strcmp(arg, "--out-device") == 0) {
				out_device_id = argv[i];
			}
			else if (strcmp(arg, "--latency") == 0) {
				microphone_latency = atof(argv[i]);
			}
			else {
				return usage(exe);
			}
		}
		else {
			return usage(exe);
		}
	}

	struct SoundIo* soundio = soundio_create();
	if (!soundio)
		panic("out of memory");

	int err = (backend == SoundIoBackendNone)
	              ? soundio_connect(soundio)
	              : soundio_connect_backend(soundio, backend);
	if (err)
		panic("error connecting: %s", soundio_strerror(err));

	soundio_flush_events(soundio);

	int default_out_device_index = soundio_default_output_device_index(soundio);
	if (default_out_device_index < 0)
		panic("no output device found");

	int default_in_device_index = soundio_default_input_device_index(soundio);
	if (default_in_device_index < 0)
		panic("no input device found");

	int in_device_index = default_in_device_index;
	if (in_device_id) {
		bool found = false;
		for (int i = 0; i < soundio_input_device_count(soundio); i += 1) {
			struct SoundIoDevice* device = soundio_get_input_device(soundio, i);
			if (device->is_raw == in_raw
			    && strcmp(device->id, in_device_id) == 0) {
				in_device_index = i;
				found = true;
				soundio_device_unref(device);
				break;
			}
			soundio_device_unref(device);
		}
		if (!found)
			panic("invalid input device id: %s", in_device_id);
	}

	int out_device_index = default_out_device_index;
	if (out_device_id) {
		bool found = false;
		for (int i = 0; i < soundio_output_device_count(soundio); i += 1) {
			struct SoundIoDevice* device
			    = soundio_get_output_device(soundio, i);
			if (device->is_raw == out_raw
			    && strcmp(device->id, out_device_id) == 0) {
				out_device_index = i;
				found = true;
				soundio_device_unref(device);
				break;
			}
			soundio_device_unref(device);
		}
		if (!found)
			panic("invalid output device id: %s", out_device_id);
	}

	struct SoundIoDevice* out_device
	    = soundio_get_output_device(soundio, out_device_index);
	if (!out_device)
		panic("could not get output device: out of memory");

	struct SoundIoDevice* in_device
	    = soundio_get_input_device(soundio, in_device_index);
	if (!in_device)
		panic("could not get input device: out of memory");

	fprintf(stderr, "Input device: %s\n", in_device->name);
	fprintf(stderr, "Output device: %s\n", out_device->name);

	int fs = 48000;
	std::cout << "HPSS hop: " << hpss_hop << std::endl;

	struct SoundIoInStream* instream = soundio_instream_create(in_device);
	if (!instream)
		panic("out of memory");
	instream->format = SoundIoFormatFloat32NE;
	instream->sample_rate = fs;
	instream->layout
	    = *soundio_channel_layout_get_builtin(SoundIoChannelLayoutIdMono);
	instream->software_latency = microphone_latency;
	instream->read_callback = read_callback;

	if ((err = soundio_instream_open(instream))) {
		fprintf(
		    stderr, "unable to open input stream: %s", soundio_strerror(err));
		return 1;
	}

	if (instream->layout_error) {
		fprintf(stderr, "unable to open input stream with layout: %s",
		        soundio_strerror(instream->layout_error));
		return 1;
	}

	struct SoundIoOutStream* outstream = soundio_outstream_create(out_device);
	if (!outstream)
		panic("out of memory");
	outstream->format = SoundIoFormatFloat32NE;
	outstream->sample_rate = fs;
	outstream->layout
	    = *soundio_channel_layout_get_builtin(SoundIoChannelLayoutIdMono);
	outstream->software_latency = microphone_latency;
	outstream->write_callback = write_callback;
	outstream->underflow_callback = underflow_callback;

	if ((err = soundio_outstream_open(outstream))) {
		fprintf(
		    stderr, "unable to open output stream: %s", soundio_strerror(err));
		return 1;
	}

	if (outstream->layout_error) {
		fprintf(stderr, "unable to open output stream with layout: %s",
		        soundio_strerror(outstream->layout_error));
		return 1;
	}

	int capacity = 2 * microphone_latency * instream->sample_rate
	               * instream->bytes_per_frame;

	hpss = new BufferedHPSS(
	    microphone_latency, capacity, hpss_hop, beta, fs, soundio);

	// execute buffered hpss in the background
	std::thread([&]() { hpss->do_hpss(); }).detach();

	char* buf = soundio_ring_buffer_write_ptr(hpss->ring_buffer_out);
	int fill_count = microphone_latency * outstream->sample_rate
	                 * outstream->bytes_per_frame;
	memset(buf, 0, fill_count);
	soundio_ring_buffer_advance_write_ptr(hpss->ring_buffer_out, fill_count);

	if ((err = soundio_instream_start(instream)))
		panic("unable to start input device: %s", soundio_strerror(err));

	if ((err = soundio_outstream_start(outstream)))
		panic("unable to start output device: %s", soundio_strerror(err));

	for (;;)
		soundio_wait_events(soundio);

	soundio_outstream_destroy(outstream);
	soundio_instream_destroy(instream);
	soundio_device_unref(in_device);
	soundio_device_unref(out_device);
	soundio_destroy(soundio);
	delete hpss;
	return 0;
}
