#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <complex>
#include <cstddef>
#include <vector>

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
		HPSS(float fs, std::size_t hop, float beta);

		// sensible defaults
		HPSS(float fs);

		~HPSS();

		void process_next_hop(std::vector<float>& next_hop);

		// recall that only the first hop samples are useful!
		std::vector<float> peek_separated_percussive();

	private:
		rhythm_toolkit_private::hpss::HPSS*
		    p_impl; // https://en.cppreference.com/w/cpp/language/pimpl
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
