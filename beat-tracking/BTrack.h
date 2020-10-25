#ifndef BTRACK_H
#define BTRACK_H

#include "CircularBuffer.h"
#include "OnsetDetection.h"
#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>

class BTrack {
private:
	static constexpr std::size_t OnsetDFBufferSize = 512;
	static constexpr std::size_t FFTLengthForACFCalculation = 1024;
	static constexpr float Tightness = 5.0F;
	static constexpr float Alpha = 0.9F;
	static constexpr float Epsilon = 0.0001F;

	int sampleRate;

	std::array<float, FFTLengthForACFCalculation> imIn // imaginaryInput = imIn
	    = {};
	std::array<float, FFTLengthForACFCalculation> realOut = {};
	std::array<float, FFTLengthForACFCalculation> imOut = {};

	int fft_order;

	IppsFFTSpec_C_32f* fft_spec;

	Ipp8u* p_mem_spec;
	Ipp8u* p_mem_init;
	Ipp8u* p_mem_buffer;

	int size_spec;
	int size_init;
	int size_buffer;

	float tempoToLagFactor;
	float beatPeriod;
	int m0;
	int beatCounter;

	int discardSamples;

	std::array<float, OnsetDFBufferSize> w1 = {};
	std::array<float, 128> w2 = {};
	std::array<float, OnsetDFBufferSize + 128> futureCumulativeScore = {};
	std::array<float, 2 * OnsetDFBufferSize> onsetDFContiguous = {};
	std::array<float, 512> acf = {};
	std::array<float, 128> combFilterBankOutput = {};
	std::array<float, 41> tempoObservationVector = {};
	std::array<float, 41> delta = {};
	std::array<float, 41> prevDelta = {};

	void processOnsetDetectionFunctionSample(float sample);
	void updateCumulativeScore(float odfSample);
	void predictBeat();
	void calculateTempo();
	void calculateOutputOfCombFilterBank();
	void calculateBalancedACF();

public:
	static constexpr std::size_t FrameSize = 1024;
	static constexpr std::size_t HopSize = 512;

	bool beatDueInFrame;
	float estimatedTempo;
	float latestCumulativeScoreValue;
	std::vector<float> currentFrameVec;
	float lastOnset;
	float* currentFrame;

	OnsetDetectionFunction odf;

	CircularBuffer<OnsetDFBufferSize> onsetDF = {};
	CircularBuffer<OnsetDFBufferSize> cumulativeScore = {};

	explicit BTrack(int sampleRate);
	~BTrack();

	void processHop(const float* samples);
};

#endif
