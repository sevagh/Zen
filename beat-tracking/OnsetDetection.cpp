#include "OnsetDetection.h"
#include "Window.h"
#include <array>
#include <cstring>
#include <iostream>
#include <math.h>

OnsetDetectionFunction::OnsetDetectionFunction()
    : fft_order(( int )log2(FrameSize))
    , p_mem_spec(nullptr)
    , p_mem_init(nullptr)
    , p_mem_buffer(nullptr)
    , size_spec(0)
    , size_init(0)
    , size_buffer(0)
{
	IppStatus ipp_status
	    = ippsFFTGetSize_C_32f(fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
	                           &size_spec, &size_init, &size_buffer);
	if (ipp_status != ippStsNoErr) {
		std::cerr << "ippFFTGetSize error: " << ipp_status << ", "
		          << ippGetStatusString(ipp_status) << std::endl;
		std::exit(-1);
	}

	for (std::size_t i = 0; i < FrameSize; ++i) {
		imIn[i] = 0.0f;
		realOut[i] = 0.0f;
		imOut[i] = 0.0f;
	}

	if (size_init > 0)
		p_mem_init = ( Ipp8u* )ippMalloc(size_init);
	if (size_buffer > 0)
		p_mem_buffer = ( Ipp8u* )ippMalloc(size_buffer);
	if (size_spec > 0)
		p_mem_spec = ( Ipp8u* )ippMalloc(size_spec);

	ipp_status = ippsFFTInit_C_32f(&fft_spec, fft_order, IPP_FFT_NODIV_BY_ANY,
	                               ippAlgHintNone, p_mem_spec, p_mem_init);
	if (ipp_status != ippStsNoErr) {
		std::cerr << "ippFFTInit error: " << ipp_status << ", "
		          << ippGetStatusString(ipp_status) << std::endl;
		std::exit(-1);
	}

	if (size_init > 0)
		ippFree(p_mem_init);
};

OnsetDetectionFunction::~OnsetDetectionFunction()
{
	if (size_buffer > 0)
		ippFree(p_mem_buffer);
	if (size_spec > 0)
		ippFree(p_mem_spec);
};

float OnsetDetectionFunction::calculate_sample(const float* buffer)
{
	// shift audio samples back in frame by hop size
	std::copy(frame.begin() + HopSize, frame.end(), frame.begin());

	// add new samples to frame from input buffer
	std::copy(buffer, buffer + HopSize, frame.end() - HopSize);

	return complex_spectral_difference_hwr();
};

void OnsetDetectionFunction::perform_FFT()
{
	size_t fsize2 = HopSize;

	for (size_t i = 0; i < fsize2; ++i) {
		std::iter_swap(frame.begin() + i, frame.begin() + i + fsize2);
		frame[i] *= window.data[i + fsize2];
		frame[i + fsize2] *= window.data[i];
	}

	ippsFFTFwd_CToC_32f(( Ipp32f* )frame.data(), ( Ipp32f* )imIn.data(),
	                    ( Ipp32f* )realOut.data(), ( Ipp32f* )imOut.data(),
	                    fft_spec, p_mem_buffer);
};

float OnsetDetectionFunction::complex_spectral_difference_hwr()
{
	float phaseDeviation;
	float sum;
	float magnitudeDifference;
	float csd;

	// perform the FFT
	perform_FFT();

	sum = 0; // initialise sum to zero

	// compute phase values from fft output and sum deviations
	for (size_t i = 0; i < FrameSize; ++i) {
		// calculate phase value
		phase[i] = atan2f(imOut[i], realOut[i]);

		// calculate magnitude value
		magSpec[i] = sqrtf(powf(realOut[i], 2) + powf(imOut[i], 2));

		// phase deviation
		phaseDeviation = phase[i] - (2 * prevPhase[i]) + prevPhase2[i];

		// calculate magnitude difference (real part of Euclidean distance
		// between complex frames)
		magnitudeDifference = magSpec[i] - prevMagSpec[i];

		// if we have a positive change in magnitude, then include in sum,
		// otherwise ignore (half-wave rectification)
		if (magnitudeDifference > 0) {
			// calculate complex spectral difference for the current spectral bin
			csd = sqrtf(powf(magSpec[i], 2) + powf(prevMagSpec[i], 2)
			            - 2 * magSpec[i] * prevMagSpec[i]
			                  * cosf(phaseDeviation));

			// add to sum
			sum = sum + csd;
		}

		// store values for next calculation
		prevPhase2[i] = prevPhase[i];
		prevPhase[i] = phase[i];
		prevMagSpec[i] = magSpec[i];
	}

	return sum;
};
