#ifndef HPSS_H
#define HPSS_H

#include <complex>
#include <cstddef>
#include <vector>
#include <thrust/device_vector.h>

#include "io.h"

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
	class HPROfflineGPU;
	class PRealtimeGPU;
};
}; // namespace rhythm_toolkit_private

namespace rhythm_toolkit {
namespace hpss {
	class HPRIOfflineGPU {
	public:
		HPRIOfflineGPU(float fs, std::size_t max_size_samples, std::size_t hop_h, std::size_t hop_p, float beta_h, float beta_p, rhythm_toolkit::io::IOGPU& io);
		HPRIOfflineGPU(float fs, std::size_t max_size_samples, std::size_t hop_h, std::size_t hop_p, rhythm_toolkit::io::IOGPU& io);
		HPRIOfflineGPU(float fs, std::size_t max_size_samples, rhythm_toolkit::io::IOGPU& io);

		~HPRIOfflineGPU();

		// copies from the io in vec, writes to the io out vec
		// pass the entire song in the in vec
		void process();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		// we use 2 cascading HPROffline objects to implement driedger's offline iterative algorithm "HPR-I"
		rhythm_toolkit_private::hpss::HPROfflineGPU *p_impl_h;
		rhythm_toolkit_private::hpss::HPROfflineGPU *p_impl_p;

		std::size_t hop_h, hop_p;
		rhythm_toolkit::io::IOGPU& io;

		// intermediate array to pass values between first and second iteration of HPSS
		thrust::device_vector<float> intermediate;
	};

	class PRealtimeGPU {
	public:
		PRealtimeGPU(float fs, std::size_t hop, float beta, rhythm_toolkit::io::IOGPU& io);
		PRealtimeGPU(float fs, std::size_t hop, rhythm_toolkit::io::IOGPU& io);
		PRealtimeGPU(float fs, rhythm_toolkit::io::IOGPU& io);

		~PRealtimeGPU();

		// copies from the io in vec, writes to the io out vec
		// pass in a real-time stream of the input, one hop at a time
		void process_next_hop();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		rhythm_toolkit_private::hpss::PRealtimeGPU *p_impl;
		rhythm_toolkit::io::IOGPU& io;
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
