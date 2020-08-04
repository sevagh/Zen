#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <cstddef>
#include <stdexcept>

namespace rhythm_toolkit {
namespace hpss {
	class HPSSException : public std::runtime_error {
	public:
		HPSSException(std::string msg)
		    : std::runtime_error(msg){};
	};

	class HPSS {
	public:
		std::size_t nwin;
		std::size_t nfft;
		std::size_t hop;
		float beta;
		window::Window win;

		HPSS(std::size_t nwin, std::size_t nfft, std::size_t hop, float beta)
		    : nwin(nwin)
		    , nfft(nfft)
		    , hop(hop)
		    , beta(beta)
		    , win(window::Window(window::WindowType::VonHann, nwin)){};

		// sensible defaults
		HPSS()
		    : HPSS(1024, 2048, 512, 2.0){};

		void do_work();
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
