#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

/*
 * Adaptation of Real-Time HPSS
 *     https://github.com/sevagh/Real-Time-HPSS
 *
 * The conclusions of Real-Time HPSS were that the percussive separation works better
 * than the harmonic. If we drop the harmonic separation completely, we don't need
 * to maintain a sliding STFT matrix, and simply do a 1D median filter on the FFT
 * before applying the IFFT.
 */
namespace rhythm_toolkit {
namespace hpss {
	class HPSSException : public std::runtime_error {
	public:
		HPSSException(std::string msg)
		    : std::runtime_error(msg){};
	};

	class HPSS {
	public:
		float fs;
		std::size_t nwin;
		std::size_t nfft;
		std::size_t hop;
		float beta;
		float eps;
		std::size_t l_harm;
		std::size_t l_perc;
		std::size_t stft_width;
		window::Window win;

		std::vector<float> last_percussive_result;
		std::vector<std::vector<float>> sliding_stft; // 1D sliding STFT matrix on which to apply median filtering and IFFT
		std::vector<std::vector<float>> s_half_mag;
		std::vector<std::vector<float>> harmonic;
		std::vector<std::vector<float>> percussive;

		HPSS(float fs, std::size_t nwin, std::size_t nfft, std::size_t hop, float beta)
		    : fs(fs)
		    , nwin(nwin)
		    , nfft(nfft)
		    , hop(hop)
		    , beta(beta)
		    , eps(std::numeric_limits<float>::epsilon())
		    , l_harm(roundf(0.2 / ((nfft - hop) / fs)))
		    , l_perc(roundf(500 / (fs / nfft)))
		    , stft_width(std::size_t(ceilf(l_harm/2)))
		    , win(window::Window(window::WindowType::VonHann, nwin))
		    , last_percussive_result(std::vector<float>(hop))
		    , sliding_stft(std::vector<std::vector<float>>(stft_width, std::vector<float>(nfft)))
		    , s_half_mag(std::vector<std::vector<float>>(stft_width, std::vector<float>(nfft)))
		    , harmonic(std::vector<std::vector<float>>(stft_width, std::vector<float>(nfft)))
		    , percussive(std::vector<std::vector<float>>(stft_width, std::vector<float>(nfft))){
			    l_perc += (1 - (l_perc % 2)); // make sure filter lengths are odd
			    l_harm += (1 - (l_harm % 2)); // make sure filter lengths are odd
		    };

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 1024, 2048, 512, 2.0){};

		void process_current_hop(const std::vector<float> &current_hop);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
