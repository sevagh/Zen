#include <soundio/soundio.h>

#include <chrono>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>

#include <libzengarden/hps.h>
#include <libzengarden/io.h>

#include <realtime.h>

static void read_callback(struct SoundIoInStream* instream,
                          int frame_count_min,
                          int frame_count_max)
{
	struct SoundIoChannelArea* areas;
	int err;

	struct SoundIoRingBuffer* ring_buffer_in
	    = reinterpret_cast<struct SoundIoRingBuffer*>(instream->userdata);

	char* write_ptr = soundio_ring_buffer_write_ptr(ring_buffer_in);
	int free_bytes = soundio_ring_buffer_free_count(ring_buffer_in);
	int free_count = free_bytes / instream->bytes_per_frame;

	if (frame_count_min > free_count) {
		std::cerr << "ring buffer overflow" << std::endl;
		std::exit(1);
	}

	int write_frames = std::min(free_count, frame_count_max);
	int frames_left = write_frames;

	for (;;) {
		int frame_count = frames_left;

		if ((err
		     = soundio_instream_begin_read(instream, &areas, &frame_count))) {
			std::cerr << "begin read error: " << soundio_strerror(err)
			          << std::endl;
			std::exit(1);
		}

		if (!frame_count)
			break;

		if (!areas) {
			// Due to an overflow there is a hole. Fill the ring buffer with
			// silence for the size of the hole.
			memset(write_ptr, 0, frame_count * instream->bytes_per_frame);
			std::cerr << "Dropped %d frames due to internal overflow\n"
			          << frame_count << std::endl;
		}
		else {
			for (int frame = 0; frame < frame_count; frame += 1) {
				memcpy(write_ptr, areas[0].ptr, instream->bytes_per_sample);
				areas[0].ptr += areas[0].step;
				write_ptr += instream->bytes_per_sample;
			}
		}

		if ((err = soundio_instream_end_read(instream))) {
			std::cerr << "end read error: " << soundio_strerror(err)
			          << std::endl;
			std::exit(1);
		}

		frames_left -= frame_count;
		if (frames_left <= 0)
			break;
	}

	int advance_bytes = write_frames * instream->bytes_per_frame;
	soundio_ring_buffer_advance_write_ptr(ring_buffer_in, advance_bytes);
}

static void write_callback(struct SoundIoOutStream* outstream,
                           int frame_count_min,
                           int frame_count_max)
{
	struct SoundIoChannelArea* areas;
	int frames_left;
	int frame_count;
	int err;

	struct SoundIoRingBuffer* ring_buffer_out
	    = reinterpret_cast<struct SoundIoRingBuffer*>(outstream->userdata);

	char* read_ptr = soundio_ring_buffer_read_ptr(ring_buffer_out);
	int fill_bytes = soundio_ring_buffer_fill_count(ring_buffer_out);
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
				std::cerr << "begin write error: " << soundio_strerror(err)
				          << std::endl;
				std::exit(1);
			}
			if (frame_count <= 0)
				return;
			for (int frame = 0; frame < frame_count; frame += 1) {
				memset(areas[0].ptr, 0, outstream->bytes_per_sample);
				areas[0].ptr += areas[0].step;
			}
			if ((err = soundio_outstream_end_write(outstream))) {
				std::cerr << "end write error: " << soundio_strerror(err)
				          << std::endl;
				std::exit(1);
			}
			frames_left -= frame_count;
		}
	}

	int read_count = std::min(frame_count_max, fill_count);
	frames_left = read_count;

	while (frames_left > 0) {
		int frame_count = frames_left;

		if ((err = soundio_outstream_begin_write(
		         outstream, &areas, &frame_count))) {
			std::cerr << "begin write error: " << soundio_strerror(err)
			          << std::endl;
			std::exit(1);
		}

		if (frame_count <= 0)
			break;

		for (int frame = 0; frame < frame_count; frame += 1) {
			memcpy(areas[0].ptr, read_ptr, outstream->bytes_per_sample);
			areas[0].ptr += areas[0].step;
			read_ptr += outstream->bytes_per_sample;
		}

		if ((err = soundio_outstream_end_write(outstream))) {
			std::cerr << "end write error: " << soundio_strerror(err)
			          << std::endl;
			std::exit(1);
		}

		frames_left -= frame_count;
	}

	soundio_ring_buffer_advance_read_ptr(
	    ring_buffer_out, read_count * outstream->bytes_per_frame);
}

static void underflow_callback(struct SoundIoOutStream* outstream)
{
	static int count = 0;
	fprintf(stderr, "underflow %d\n", ++count);
}

int zg::realtime::RealtimeCommand::init()
{
	soundio = soundio_create();
	if (!soundio) {
		std::cerr << "out of memory" << std::endl;
		return 1;
	}

	int err = soundio_connect(soundio);
	if (err) {
		std::cerr << "error connecting: " << soundio_strerror(err) << std::endl;
		return 1;
	}

	soundio_flush_events(soundio);

	int default_out_device_index = soundio_default_output_device_index(soundio);
	if (default_out_device_index < 0) {
		std::cerr << "no output device found" << std::endl;
		return 1;
	}

	int default_in_device_index = soundio_default_input_device_index(soundio);
	if (default_in_device_index < 0) {
		std::cerr << "no input device found" << std::endl;
		return 1;
	}

	int in_device_index = default_in_device_index;
	if (p.indevice != "") {
		bool found = false;
		for (int i = 0; i < soundio_input_device_count(soundio); i += 1) {
			struct SoundIoDevice* device = soundio_get_input_device(soundio, i);
			if (strcmp(device->id, p.indevice.c_str()) == 0) {
				in_device_index = i;
				found = true;
				soundio_device_unref(device);
				break;
			}
			soundio_device_unref(device);
		}
		if (!found) {
			std::cerr << "invalid input device id: " << p.indevice << std::endl;
			return 1;
		}
	}

	int out_device_index = default_out_device_index;
	if (p.outdevice != "") {
		bool found = false;
		for (int i = 0; i < soundio_output_device_count(soundio); i += 1) {
			struct SoundIoDevice* device
			    = soundio_get_output_device(soundio, i);
			if (strcmp(device->id, p.outdevice.c_str()) == 0) {
				out_device_index = i;
				found = true;
				soundio_device_unref(device);
				break;
			}
			soundio_device_unref(device);
		}
		if (!found) {
			std::cerr << "invalid output device id: " << p.outdevice
			          << std::endl;
			return 1;
		}
	}

	out_device = soundio_get_output_device(soundio, out_device_index);
	if (!out_device) {
		std::cerr << "could not get output device: out of memory" << std::endl;
		return 1;
	}

	in_device = soundio_get_input_device(soundio, in_device_index);
	if (!in_device) {
		std::cerr << "could not get input device: out of memory" << std::endl;
		return 1;
	}

	std::cerr << "Input device: " << in_device->name << std::endl;
	;
	std::cerr << "Output device: " << out_device->name << std::endl;
	;

	instream = soundio_instream_create(in_device);
	if (!instream) {
		std::cerr << "out of memory" << std::endl;
		return 1;
	}

	instream->format = SoundIoFormatFloat32NE;
	instream->sample_rate = p.fs;
	instream->layout
	    = *soundio_channel_layout_get_builtin(SoundIoChannelLayoutIdMono);
	instream->software_latency = p.microphone_latency;
	instream->read_callback = read_callback;

	if ((err = soundio_instream_open(instream))) {
		std::cerr << "unable to open input stream: " << soundio_strerror(err)
		          << std::endl;
		return 1;
	}

	if (instream->layout_error) {
		std::cerr << "unable to open input stream layout: "
		          << soundio_strerror(instream->layout_error) << std::endl;
		return 1;
	}

	outstream = soundio_outstream_create(out_device);
	if (!outstream) {
		std::cerr << "out of memory" << std::endl;
		return 1;
	}
	outstream->format = SoundIoFormatFloat32NE;
	outstream->sample_rate = p.fs;
	outstream->layout
	    = *soundio_channel_layout_get_builtin(SoundIoChannelLayoutIdMono);
	outstream->software_latency = p.microphone_latency;
	outstream->write_callback = write_callback;
	outstream->underflow_callback = underflow_callback;

	if ((err = soundio_outstream_open(outstream))) {
		std::cerr << "unable to open output stream: " << soundio_strerror(err)
		          << std::endl;
		return 1;
	}

	if (outstream->layout_error) {
		std::cerr << "unable to open output stream layout: "
		          << soundio_strerror(outstream->layout_error) << std::endl;
		return 1;
	}

	int capacity = 2 * p.microphone_latency * instream->sample_rate
	               * instream->bytes_per_frame;

	bloop = new BufferedLoop(soundio, capacity, p);

	instream->userdata = reinterpret_cast<void*>(bloop->ring_buffer_in);
	outstream->userdata = reinterpret_cast<void*>(bloop->ring_buffer_out);

	return 0;
}

int zg::realtime::RealtimeCommand::execute()
{
	int err;

	// execute buffered loop in the background
	std::thread([&]() { bloop->execute_effects_loop(); }).detach();

	char* buf = soundio_ring_buffer_write_ptr(bloop->ring_buffer_out);
	int fill_count = p.microphone_latency * outstream->sample_rate
	                 * outstream->bytes_per_frame;
	memset(buf, 0, fill_count);
	soundio_ring_buffer_advance_write_ptr(bloop->ring_buffer_out, fill_count);

	if ((err = soundio_instream_start(instream))) {
		std::cerr << "unable to start input device: " << soundio_strerror(err)
		          << std::endl;
		return 1;
	}

	if ((err = soundio_outstream_start(outstream))) {
		std::cerr << "unable to start output device: " << soundio_strerror(err)
		          << std::endl;
		return 1;
	}

	for (;;)
		soundio_wait_events(soundio);
}
