#ifndef HPSS_H
#define HPSS_H

#include "window.h"
#include <cstddef>
#include <stdexcept>
#include <opencv2/core/mat.hpp>

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
		std::size_t l_perc;
		window::Window win;

		std::vector<float> last_percussive_result;
		cv::Mat sliding_stft; // 1D STFT "matrix" on which to apply median filtering and IFFT
		cv::Mat s_half_mag; // to always store the half magnitude of the stft here
		cv::Mat percussive;

		HPSS(float fs, std::size_t nwin, std::size_t nfft, std::size_t hop, float beta)
		    : fs(fs)
		    , nwin(nwin)
		    , nfft(nfft)
		    , hop(hop)
		    , beta(beta)
		    , eps(std::numeric_limits<float>::epsilon())
		    , l_perc(roundf(500 / (fs / nfft)))
		    , win(window::Window(window::WindowType::VonHann, nwin))
		    , last_percussive_result(std::vector<float>(hop))
		    , sliding_stft(cv::Mat::zeros(1, nfft, CV_32F))
		    , s_half_mag(cv::Mat::zeros(1, nfft, CV_32F))
		    , percussive(cv::Mat::zeros(1, nfft, CV_32F)){
			    l_perc += (1 - (l_perc % 2)); // make sure it's odd
		    };

		// sensible defaults
		HPSS(float fs)
		    : HPSS(fs, 1024, 2048, 512, 2.0){};

		void process_current_hop(const std::vector<float> &current_hop);
	};
}; // namespace hpss
}; // namespace rhythm_toolkit

#endif /* HPSS_H */
