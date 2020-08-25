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
	class HPSS;
};
}; // namespace rhythm_toolkit_private

namespace rhythm_toolkit {
namespace hpss {
	class HPSS {
	public:
		HPSS(float fs, std::size_t hop_h, std::size_t hop_p, float beta_h, float beta_p, rhythm_toolkit::io::IO& io);

		HPSS(float fs, std::size_t hop_h, std::size_t hop_p, rhythm_toolkit::io::IO& io);

		HPSS(float fs, rhythm_toolkit::io::IO& io);

		~HPSS();

		// copies from the io in vec, writes to the io out vec
		void process_next_hop();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		// we use 2 cascading HPSS objects to implement driedger's iterative algorithm "HPR-I"
		rhythm_toolkit_private::hpss::HPSS *p_impl_h;
		rhythm_toolkit_private::hpss::HPSS *p_impl_p;

		std::size_t hop_h, hop_p;
		rhythm_toolkit::io::IO& io;

		// intermediate array to pass values between first and second iteration of HPSS
		thrust::device_vector<float> intermediate;
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
