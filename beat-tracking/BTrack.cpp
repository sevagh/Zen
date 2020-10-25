#include "BTrack.h"
#include "BTrackPrecomputed.h"
#include "CircularBuffer.h"
#include "OnsetDetection.h"
#include <array>
#include <complex>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <vector>

static float
calculateMeanOfArray(const float* array, std::size_t start, std::size_t end);
static void adaptiveThreshold(float* x, size_t N);
static void normalizeArray(float* x, size_t N);
static void createWindow(std::array<float, 512>& w1,
                         std::size_t start,
                         std::size_t end,
                         float beatPeriod,
                         float Tightness);

BTrack::BTrack(int sampleRate)
    : sampleRate(sampleRate)
    , tempoToLagFactor(60.0F * (( float )sampleRate) / ( float )HopSize)
    , latestCumulativeScoreValue(0.0F)
    , lastOnset(0.0F)
    , odf(OnsetDetectionFunction())
    , beatPeriod(
          roundf(60.0F / (((( float )HopSize) / ( float )sampleRate) * 120.0F)))
    , m0(10)
    , beatCounter(-1)
    , discardSamples(sampleRate / 2)
    , beatDueInFrame(false)
    , estimatedTempo(120.0F)
    , fft_order(( int )log2(FrameSize))
    , p_mem_spec(nullptr)
    , p_mem_init(nullptr)
    , p_mem_buffer(nullptr)
    , size_spec(0)
    , size_init(0)
    , size_buffer(0)
{
	std::fill(prevDelta.begin(), prevDelta.end(), 1.0F);

	// initialise df_buffer
	for (size_t i = 0; i < OnsetDFBufferSize; ++i) {
		if ((i % (( size_t )round(beatPeriod))) == 0) { // TODO faster modulo
			onsetDF[i] = 1.0F;
		}
	}

	// first r2c fft
	IppStatus ipp_status
	    = ippsFFTGetSize_C_32f(fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
	                           &size_spec, &size_init, &size_buffer);
	if (ipp_status != ippStsNoErr) {
		std::cerr << "ippFFTGetSize error: " << ipp_status << ", "
		          << ippGetStatusString(ipp_status) << std::endl;
		std::exit(-1);
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

	for (std::size_t i = 0; i < FrameSize; ++i) {
		imIn[i] = 0.0f;
	}
};

BTrack::~BTrack()
{
	if (size_buffer > 0)
		ippFree(p_mem_buffer);
	if (size_spec > 0)
		ippFree(p_mem_spec);
};

void BTrack::processHop(const float* samples)
{
	float sample = odf.calculate_sample(samples);
	lastOnset = sample;
	processOnsetDetectionFunctionSample(sample);
}

void BTrack::processOnsetDetectionFunctionSample(float sample)
{
	sample = fabs(sample) + Epsilon;
	m0--;
	beatCounter--;
	beatDueInFrame = false;

	onsetDF.append(sample);
	updateCumulativeScore(sample);

	if (m0 == 0) {
		predictBeat();
	}

	if (beatCounter == 0) {
		beatDueInFrame = true;
		calculateTempo();
	}
};

void BTrack::updateCumulativeScore(float odfSample)
{
	auto start = (size_t)(OnsetDFBufferSize - roundf(2.0F * beatPeriod));
	auto end = (size_t)(OnsetDFBufferSize - roundf(beatPeriod / 2.0F));

	createWindow(w1, start, end, beatPeriod, Tightness);

	float max = 0.0F;
	for (size_t i = start; i <= end; ++i) {
		max = std::max(cumulativeScore[i] * w1[i - start], max);
	}

	latestCumulativeScoreValue = ((1.0F - Alpha) * odfSample) + (Alpha * max);
	cumulativeScore.append(latestCumulativeScoreValue);
};

void BTrack::predictBeat()
{
	auto windowSize = ( size_t )beatPeriod;

	// copy cumscore to first part of fcumscore
	for (size_t i = 0; i < OnsetDFBufferSize; ++i) {
		futureCumulativeScore[i] = cumulativeScore[i];
	}

	// create future window - TODO can this be precomputed?
	float v = 1.0F;
	for (size_t i = 0; i < windowSize; ++i) {
		w2[i] = expf((-1.0F * powf((v - (beatPeriod / 2.0F)), 2.0F))
		             / (2.0F * powf((beatPeriod / 2.0F), 2.0F)));
		v += 1.0F;
	}

	// calculate future cumulative score
	float max;
	int n;
	float wcumscore;
	for (size_t i = OnsetDFBufferSize; i < (OnsetDFBufferSize + windowSize);
	     ++i) {
		auto start = (size_t)(i - roundf(2.0F * beatPeriod));
		auto end = (size_t)(i - roundf(beatPeriod / 2.0F));

		max = 0;
		n = 0;
		for (size_t k = start; k <= end; ++k) {
			// use the same w1 calculated previously
			wcumscore = futureCumulativeScore[k] * w1[n];

			if (wcumscore > max) {
				max = wcumscore;
			}
			n++;
		}

		futureCumulativeScore[i] = max;
	}

	// predict beat
	max = 0;
	n = 0;

	for (size_t i = OnsetDFBufferSize; i < (OnsetDFBufferSize + windowSize);
	     ++i) {
		wcumscore = futureCumulativeScore[i] * w2[n];

		if (wcumscore > max) {
			max = wcumscore;
			beatCounter = n;
		}

		n++;
	}

	m0 = ( int )(beatCounter + roundf(beatPeriod / 2.0F));
};

void BTrack::calculateTempo()
{
	for (size_t i = 0; i < OnsetDFBufferSize; ++i) {
		onsetDFContiguous[i] = onsetDF[i];
	}

	// adaptive threshold on input - TODO faster math with MathNeon
	adaptiveThreshold(onsetDFContiguous.data(), onsetDFContiguous.size());

	// calculate auto-correlation function of detection function
	calculateBalancedACF();

	// calculate output of comb filterbank - TODO faster math with MathNeon
	calculateOutputOfCombFilterBank();

	// adaptive threshold on rcf
	adaptiveThreshold(combFilterBankOutput.data(), combFilterBankOutput.size());

	size_t t_index;
	size_t t_index2;
	// calculate tempo observation vector from beat period observation vector
	for (size_t i = 0; i < 41; ++i) {
		t_index = ( size_t )roundf(tempoToLagFactor / (((2.0F * i) + 80.0F)));
		t_index2 = t_index / 2;

		tempoObservationVector[i] = combFilterBankOutput[t_index - 1]
		                            + combFilterBankOutput[t_index2 - 1];
	}

	float maxind;

	float maxval;
	for (size_t j = 0; j < 41; ++j) {
		maxval = -1.0F;
		for (size_t i = 0; i < 41; ++i)
			maxval = std::max(
			    prevDelta[i] * precomputed::TempoTransitionMatrix[i][j],
			    maxval);

		delta[j] = maxval * tempoObservationVector[j];
	}

	normalizeArray(delta.data(), 41);

	maxind = -1;
	maxval = -1;

	for (size_t j = 0; j < 41; ++j) {
		if (delta[j] > maxval) {
			maxval = delta[j];
			maxind = j;
		}

		prevDelta[j] = delta[j];
	}

	beatPeriod = roundf((60.0F * (( float )sampleRate))
	                    / (((2.0F * maxind) + 80.0F) * (( float )HopSize)));

	if (beatPeriod > 0) {
		estimatedTempo
		    = 60.0F
		      / (((( float )HopSize) / (( float )sampleRate)) * beatPeriod);
	}
};

void BTrack::calculateOutputOfCombFilterBank()
{
	std::fill(combFilterBankOutput.begin(), combFilterBankOutput.end(), 0.0F);

	for (int i = 2; i <= 127; ++i) {   // max beat period
		for (int a = 1; a <= 4; ++a) { // number of comb elements
			for (int b = 1 - a; b <= a - 1;
			     ++b) { // general state using normalisation of
				        // comb elements
				combFilterBankOutput[i - 1]
				    = combFilterBankOutput[i - 1]
				      + (acf[(a * i + b) - 1]
				         * precomputed::RayleighWeightingVector128[i - 1])
				            / (2 * a - 1); // calculate value for comb
				                           // filter row
			}
		}
	}
};

void BTrack::calculateBalancedACF()
{
	// the first 512 samples already contain the onset detection values
	// zero pad the remaining 512
	std::fill(onsetDFContiguous.begin() + 512, onsetDFContiguous.end(), 0.0F);

	ippsFFTFwd_CToC_32f(( Ipp32f* )onsetDFContiguous.data(),
	                    ( Ipp32f* )imIn.data(), ( Ipp32f* )realOut.data(),
	                    ( Ipp32f* )imOut.data(), fft_spec, p_mem_buffer);

	// multiply by complex conjugate
	for (int i = 0; i < FFTLengthForACFCalculation; i++) {
		realOut[i] = realOut[i] * realOut[i] + imOut[i] * imOut[i];
		imOut[i] = 0.0F;
	}

	ippsFFTInv_CToC_32f_I(( Ipp32f* )realOut.data(), ( Ipp32f* )imOut.data(),
	                      fft_spec, p_mem_buffer);

	for (size_t i = 0; i < OnsetDFBufferSize; i++) {
		acf[i] = sqrtf(realOut[i] * realOut[i] + imOut[i] * imOut[i])
		         / ( float )(OnsetDFBufferSize - i);
	}
};

static void normalizeArray(float* x, size_t N)
{
	float sum = 0.0F;

	for (size_t i = 0; i < N; i++) {
		if (x[i] > 0) {
			sum += x[i];
		}
	}

	if (sum > 0) {
		for (size_t i = 0; i < N; i++) {
			x[i] /= sum;
		}
	}
};

// we only call adapativeThreshold on two arrays - N = 1024, N = 128
static std::array<float, 1024> x_thresh = {};

static void adaptiveThreshold(float* x, size_t N)
{
	std::size_t i = 0;
	std::size_t k = 0;
	std::size_t t = 0;

	std::size_t p_post = 7;
	std::size_t p_pre = 8;

	t = std::min(N, p_post); // what is smaller, p_post or df size. This is to
	// avoid accessing outside of arrays

	// find threshold for first 't' samples, where a full average cannot be
	// computed yet
	for (i = 0; i <= t; ++i) {
		k = std::min((i + p_pre), N);
		x_thresh[i] = calculateMeanOfArray(x, 1, k);
	}
	// find threshold for bulk of samples across a moving average from
	// [i-p_pre,i+p_post]
	for (i = t + 1; i < N - p_post; ++i) {
		x_thresh[i] = calculateMeanOfArray(x, i - p_pre, i + p_post);
	}

	// for last few samples calculate threshold, again, not enough samples
	// to do as above
	for (i = N - p_post; i < N; ++i) {
		k = std::max<std::size_t>(i - p_post, 1);
		x_thresh[i] = calculateMeanOfArray(x, k, N);
	}

	// subtract the threshold from the detection function and check that it
	// is not less than 0
	for (i = 0; i < N; ++i) {
		x[i] = x[i] - x_thresh[i];
		if (x[i] < 0) {
			x[i] = 0;
		}
	}
};

static float
calculateMeanOfArray(const float* array, std::size_t start, std::size_t end)
{
	float sum = 0;
	std::size_t length = end - start;

	// find sum
	for (std::size_t i = start; i < end; i++) {
		sum = sum + array[i];
	}

	return (length > 0) ? sum / length : 0;
}

static void createWindow(std::array<float, 512>& w1,
                         std::size_t start,
                         std::size_t end,
                         float beatPeriod,
                         float Tightness)
{
	float v = -2.0F * beatPeriod;
	std::size_t winsize = end - start + 1;
	// create window
	for (size_t i = 0; i < winsize; ++i) {
		// TODO replace with faster MathNeon computations
		w1[i]
		    = expf((-1 * powf(Tightness * logf(-v / beatPeriod), 2.0F)) / 2.0F);
		v += 1.0F;
	}
}
