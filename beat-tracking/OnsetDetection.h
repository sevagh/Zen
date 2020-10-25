#ifndef BTRACK_ONSETDETECTION_H
#define BTRACK_ONSETDETECTION_H

#include "Window.h"
#include <array>
#include <cstddef>
#include <vector>

#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>

class OnsetDetectionFunction {
public:
	static constexpr std::size_t FrameSize = 1024;
	std::array<float, FrameSize> imIn = {};
	std::array<float, FrameSize> realOut = {};
	std::array<float, FrameSize> imOut = {};

	OnsetDetectionFunction();
	~OnsetDetectionFunction();

	float calculate_sample(const float* buffer);

private:
	static constexpr std::size_t HopSize = 512;
	static constexpr WindowType windowType = HanningWindow;
	void perform_FFT();
	float complex_spectral_difference_hwr();

	// compute windows at compile time
	Window<FrameSize> window = get_window<FrameSize, windowType>();

	int fft_order;

	IppsFFTSpec_C_32f* fft_spec;

	Ipp8u* p_mem_spec;
	Ipp8u* p_mem_init;
	Ipp8u* p_mem_buffer;

	int size_spec;
	int size_init;
	int size_buffer;

	std::array<float, FrameSize> frame = {};
	std::array<float, FrameSize> magSpec = {};
	std::array<float, FrameSize> prevMagSpec = {};
	std::array<float, FrameSize> phase = {};
	std::array<float, FrameSize> prevPhase = {};
	std::array<float, FrameSize> prevPhase2 = {};
};

#endif
