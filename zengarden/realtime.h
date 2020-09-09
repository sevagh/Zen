#ifndef ZG_CLI_REALTIME
#define ZG_CLI_REALTIME

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

namespace zg {
namespace realtime {

	struct RealtimeParams {
		std::string indevice = "";
		std::string outdevice = "";
		bool do_hps = false;
		std::size_t hop = 256;
		float beta = 2.5;
		int fs = 48000;
		float microphone_latency = 0.02;
	};

	class BufferedLoop {
	public:
		struct SoundIoRingBuffer* ring_buffer_in;
		struct SoundIoRingBuffer* ring_buffer_out;

		BufferedLoop(struct SoundIo* soundio,
		             std::size_t capacity,
		             RealtimeParams p)
		    : p(p)
		    , io(zg::io::IOGPU(p.hop))
		    , hpss(zg::hps::PRealtimeGPU(( float )p.fs, p.hop, p.beta, io))
		{

			ring_buffer_in = soundio_ring_buffer_create(soundio, capacity);
			if (!ring_buffer_in) {
				std::cerr << "unable to create ring buffer in: out of memory"
				          << std::endl;
				std::exit(1);
			}

			ring_buffer_out = soundio_ring_buffer_create(soundio, capacity);
			if (!ring_buffer_out) {
				std::cerr << "unable to create ring buffer out: out of memory"
				          << std::endl;
				std::exit(1);
			}

			if (p.do_hps) {
				std::cout << "warming up HPSS first..." << std::endl;
				hpss.warmup();
			}
		}

		~BufferedLoop()
		{
			soundio_ring_buffer_destroy(ring_buffer_in);
			soundio_ring_buffer_destroy(ring_buffer_out);
		}

		void execute_effects_loop()
		{
			for (;;) {
				int input_count = soundio_ring_buffer_fill_count(ring_buffer_in)
				                  / sizeof(float);
				int output_free_count
				    = soundio_ring_buffer_free_count(ring_buffer_out)
				      / sizeof(float);

				// if there are at least hop samples to write to the hpss
				// object and at least hop space in the output ringbuffer in
				// which to write the results
				if (input_count >= p.hop && output_free_count >= p.hop) {
					std::size_t rw_amount = p.hop * sizeof(float);

					char* read_ptr
					    = soundio_ring_buffer_read_ptr(ring_buffer_in);
					char* write_ptr
					    = soundio_ring_buffer_write_ptr(ring_buffer_out);

					if (p.do_hps) {
						memcpy(( char* )io.host_in, read_ptr, rw_amount);

						hpss.process_next_hop();

						auto percussive_limits = std::minmax_element(
						    io.host_out, io.host_out + p.hop);

						float real_perc_max
						    = std::max(-1 * (*percussive_limits.first),
						               *percussive_limits.second);

						// normalize between -1.0 and 1.0
						for (std::size_t i = 0; i < p.hop; ++i) {
							io.host_out[i] /= real_perc_max;
						}

						memcpy(write_ptr, ( char* )io.host_out, rw_amount);
					}
					else {
						// straight loopback of input to output
						memcpy(write_ptr, read_ptr, rw_amount);
					}

					soundio_ring_buffer_advance_read_ptr(
					    ring_buffer_in, rw_amount);

					soundio_ring_buffer_advance_write_ptr(
					    ring_buffer_out, rw_amount);

					// get input count for next iteration
					input_count
					    = soundio_ring_buffer_fill_count(ring_buffer_in);
				}
			}
		}

	private:
		zg::io::IOGPU io;
		zg::hps::PRealtimeGPU hpss;
		RealtimeParams p;
	};

	class RealtimeCommand {
	public:
		RealtimeCommand(RealtimeParams p)
		    : p(p){};

		int init();

		int execute();

		~RealtimeCommand()
		{
			soundio_outstream_destroy(outstream);
			soundio_instream_destroy(instream);
			soundio_device_unref(in_device);
			soundio_device_unref(out_device);
			soundio_destroy(soundio);
		}

	private:
		RealtimeParams p;
		struct SoundIo* soundio;
		struct SoundIoDevice* in_device;
		struct SoundIoDevice* out_device;
		struct SoundIoInStream* instream;
		struct SoundIoOutStream* outstream;
		BufferedLoop* bloop;
	};
}; // namespace realtime
}; // namespace zg

#endif /* ZG_CLI_OFFLINE */
