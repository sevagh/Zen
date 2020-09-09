#include <onset.h>
#include "NE10.h"
#include "Window.h"
#include <array>
#include <cmath>

namespace zg {
namespace onset {

OnsetDetectionFunction::OnsetDetectionFunction()
    : fftCfg(ne10_fft_alloc_r2c_float32(FrameSize))
{
};

OnsetDetectionFunction::~OnsetDetectionFunction()
{
	ne10_fft_destroy_r2c_float32(fftCfg);
};

float OnsetDetectionFunction::calculate_sample(std::vector<float>& buffer)
{
	// shift audio samples back in frame by hop size
	std::copy(frame.begin() + HopSize, frame.end(), frame.begin());

	// add new samples to frame from input buffer
	std::copy(buffer.begin(), buffer.begin() + HopSize, frame.end() - HopSize);

	// compute complex spectral difference HWR
	float phaseDeviation;
	float sum;
	float magnitudeDifference;
	float csd;

	// perform the FFT
	size_t fsize2 = HopSize;

	for (size_t i = 0; i < fsize2; ++i) {
		std::iter_swap(frame.begin() + i, frame.begin() + i + fsize2);
		frame[i] *= window.window[i + fsize2];
		frame[i + fsize2] *= window.window[i];
	}

	ne10_fft_r2c_1d_float32_neon(complexOut.data(), frame.data(), fftCfg);

	sum = 0; // initialise sum to zero

	// compute phase values from fft output and sum deviations
	for (size_t i = 0; i < FrameSize; ++i) {
		// calculate phase value
		phase[i] = std::atan2f(complexOut[i].i, complexOut[i].r);

		// calculate magnitude value
		magSpec[i] = std::sqrtf(std::powf(complexOut[i].r, 2)
		                        + std::powf(complexOut[i].i, 2));

		// phase deviation
		phaseDeviation = phase[i] - (2 * prevPhase[i]) + prevPhase2[i];

		// calculate magnitude difference (real part of Euclidean distance
		// between complex frames)
		magnitudeDifference = magSpec[i] - prevMagSpec[i];

		// if we have a positive change in magnitude, then include in sum,
		// otherwise ignore (half-wave rectification)
		if (magnitudeDifference > 0) {
			// calculate complex spectral difference for the current spectral bin
			csd = std::sqrtf(
			    std::powf(magSpec[i], 2) + std::powf(prevMagSpec[i], 2)
			    - 2 * magSpec[i] * prevMagSpec[i] * std::cosf(phaseDeviation));

			// add to sum
			sum = sum + csd;
		}

		// store values for next calculation
		prevPhase2[i] = prevPhase[i];
		prevPhase[i] = phase[i];
		prevMagSpec[i] = magSpec[i];
	}

	memcpy(magSpecCopy, magSpec.data(), (FrameSize / 2) * sizeof(float));
	return sum;
};
} // namespace btrack
