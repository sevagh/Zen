#ifndef PITCH_DETECTION_H
#define PITCH_DETECTION_H

#include <complex>
#include <ffts/ffts.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>

class MPM {
public:
	long N;
	float sample_rate;
	std::vector<std::complex<float>> out_im;
	std::vector<float> out_real;

	int fft_order;

	IppsFFTSpec_C_32fc* fft_spec;

	Ipp8u* p_mem_spec;
	Ipp8u* p_mem_init;
	Ipp8u* p_mem_buffer;

	int size_spec;
	int size_init;
	int size_buffer;

	MPM(long audio_buffer_size, float sample_rate)
	    : N(audio_buffer_size)
	    , sample_rate(sample_rate)
	    , out_im(std::vector<std::complex<float>>(N * 2))
	    , out_real(std::vector<float>(N))
	    , fft_order(( int )log2(2 * N))
	    , p_mem_spec(nullptr)
	    , p_mem_init(nullptr)
	    , p_mem_buffer(nullptr)
	    , size_spec(0)
	    , size_init(0)
	    , size_buffer(0)
	{
		if (N == 0) {
			throw std::bad_alloc();
		}

		IppStatus ipp_status = ippsFFTGetSize_C_32fc(
		    fft_order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &size_spec,
		    &size_init, &size_buffer);
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

		ipp_status
		    = ippsFFTInit_C_32fc(&fft_spec, fft_order, IPP_FFT_NODIV_BY_ANY,
		                         ippAlgHintNone, p_mem_spec, p_mem_init);
		if (ipp_status != ippStsNoErr) {
			std::cerr << "ippFFTInit error: " << ipp_status << ", "
			          << ippGetStatusString(ipp_status) << std::endl;
			std::exit(-1);
		}

		if (size_init > 0)
			ippFree(p_mem_init);
	}

	float pitch(const float*);

	~MPM()
	{
		if (size_buffer > 0)
			ippFree(p_mem_buffer);
		if (size_spec > 0)
			ippFree(p_mem_spec);
	}

protected:
	void clear()
	{
		std::fill(out_im.begin(), out_im.end(), std::complex<float>(0.0, 0.0));
	}
};

#endif
