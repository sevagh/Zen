#ifndef ZG_HPS_PUB_H
#define ZG_HPS_PUB_H

#include <complex>
#include <cstddef>
#include <thrust/device_vector.h>
#include <vector>

#include <libzengarden/io.h>

namespace zg {
namespace hps {
	// forward declare the private implementations of HPSS
	class HPRGPU;
	class HPRCPU;

	class HPRIOfflineGPU {
	public:
		HPRIOfflineGPU(float fs,
		               std::size_t hop_h,
		               std::size_t hop_p,
		               float beta_h,
		               float beta_p);
		HPRIOfflineGPU(float fs, std::size_t hop_h, std::size_t hop_p);
		HPRIOfflineGPU(float fs);

		~HPRIOfflineGPU();

		// pass the entire song in the in vec
		// the vector is copied to be modified in-place
		//
		// returns a triplet of harmonic,percussive,residual results
		// of audio.size()
		std::vector<float> process(std::vector<float> audio);

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		// we use 2 cascading HPR objects to implement driedger's
		// offline iterative algorithm "HPR-I"
		zg::hps::HPRGPU* p_impl_h;
		zg::hps::HPRGPU* p_impl_p;

		std::size_t hop_h, hop_p;
		zg::io::IOGPU io_h;
		zg::io::IOGPU io_p;
	};

	class HPRIOfflineCPU {
	public:
		HPRIOfflineCPU(float fs,
		               std::size_t hop_h,
		               std::size_t hop_p,
		               float beta_h,
		               float beta_p);
		HPRIOfflineCPU(float fs, std::size_t hop_h, std::size_t hop_p);
		HPRIOfflineCPU(float fs);

		~HPRIOfflineCPU();

		// pass the entire song in the in vec
		// the vector is copied to be modified in-place
		//
		// returns a triplet of harmonic,percussive,residual results
		// of audio.size()
		std::vector<float> process(std::vector<float> audio);

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		// we use 2 cascading HPR objects to implement driedger's
		// offline iterative algorithm "HPR-I"
		zg::hps::HPRCPU* p_impl_h;
		zg::hps::HPRCPU* p_impl_p;

		std::size_t hop_h, hop_p;
	};

	class PRealtimeGPU {
	public:
		PRealtimeGPU(float fs,
		             std::size_t hop,
		             float beta,
		             zg::io::IOGPU& io);
		PRealtimeGPU(float fs, std::size_t hop, zg::io::IOGPU& io);
		PRealtimeGPU(float fs, zg::io::IOGPU& io);

		~PRealtimeGPU();

		// copies from the io in vec, writes to the io out vec
		// pass in a real-time stream of the input, one hop at a time
		void process_next_hop();

		// gpu benefits from some warmup, especially in real-time contexts
		// where the slow early iterations cause latency issues
		void warmup();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		zg::hps::HPRGPU* p_impl;
		zg::io::IOGPU& io;
	};
}; // namespace hps
}; // namespace zg

#endif /* ZG_HPS_PUB_H */
