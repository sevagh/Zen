#ifndef ZG_ONSET_H
#define ZG_ONSET_H

#include <array>
#include <cmath>
#include <cstddef>
#include <fftw.h>
#include <vector>
#include <win.h>

namespace zg {
namespace onset {
	class OnsetDetectionCPU {
	public:
		static constexpr std::size_t FrameSize = 1024;
		std::array<ne10_fft_cpx_float32_t, FrameSize> complexOut = {};

		OnsetDetectionFunction(std::size_t hop)
		    : hop(hop)
		    , nwin(2 * hop)
		    , window(zg::win::WindowType::VonHann, nwin){};

		~OnsetDetectionFunction();

		float calculate_sample(std::vector<float>& buffer);

	private:
		std::size_t hop;
		std::size_t nwin;
		zg::win::WindowCPU window;

		zg::fftw::FFTWrapperCPU fft;

		std::vector<float> frame;
		std::vector<float> magSpec;
		std::vector<float> prevMagSpec;
		std::vector<float> phase;
		std::vector<float> prevPhase;
		std::vector<float> prevPhase2;
	};
} // namespace onset

#endif /* ZG_ONSET_H */
