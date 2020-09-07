#ifndef HPSS_H
#define HPSS_H

#include <complex>
#include <cstddef>
#include <thrust/device_vector.h>
#include <vector>

#include "io.h"
#include "rhythm_toolkit.h"

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 */

// forward declare the private implementation of HPSS
namespace rhythm_toolkit_private {
namespace hpss {
	class HPRGPU;
	class HPRCPU;
}; // namespace hpss
}; // namespace rhythm_toolkit_private

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works
 * better than the harmonic. If we drop the harmonic separation completely, we
 * save some computation (although the harmonic mask is necessary for a
 * percussive separation)
 *
 * 2 classes:
 *
 * "HPRIOffline" - Harmonic-Percussive-Residual Iterative Offline
 * uses past, present, and future frames for optimal separation of harmonic
 * uses a large block size for harmonic separation followed by a small block
 * size for percussive
 *
 * "PRealtime" - Percussive Realtime
 * this one only uses present and past frames to respect causality for
 * real-time it omits the computation of the harmonic and residual
 */
namespace rhythm_toolkit {
namespace hpss {
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
		rhythm_toolkit_private::hpss::HPRGPU* p_impl_h;
		rhythm_toolkit_private::hpss::HPRGPU* p_impl_p;

		std::size_t hop_h, hop_p;
		rhythm_toolkit::io::IOGPU io_h;
		rhythm_toolkit::io::IOGPU io_p;
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
		rhythm_toolkit_private::hpss::HPRCPU* p_impl_h;
		rhythm_toolkit_private::hpss::HPRCPU* p_impl_p;

		std::size_t hop_h, hop_p;
	};

	class PRealtimeGPU {
	public:
		PRealtimeGPU(float fs,
		             std::size_t hop,
		             float beta,
		             rhythm_toolkit::io::IOGPU& io);
		PRealtimeGPU(float fs, std::size_t hop, rhythm_toolkit::io::IOGPU& io);
		PRealtimeGPU(float fs, rhythm_toolkit::io::IOGPU& io);

		~PRealtimeGPU();

		// copies from the io in vec, writes to the io out vec
		// pass in a real-time stream of the input, one hop at a time
		void process_next_hop();

		// gpu benefits from some warmup, especially in real-time contexts
		// where the slow early iterations cause latency issues
		void warmup();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		rhythm_toolkit_private::hpss::HPRGPU* p_impl;
		rhythm_toolkit::io::IOGPU& io;
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
