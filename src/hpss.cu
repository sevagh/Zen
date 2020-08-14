#include <stdio.h>
#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <iostream>
#include "rhythm_toolkit_priv.h"

void cuda_hpss(thrust::host_vector<float>& x_, thrust::host_vector<float>& p_)
{
	thrust::device_vector<float> x(x_);
	thrust::device_vector<thrust::complex<float>> fft(x.size()/2 + 1);

	cufftHandle plan_forward;
	cufftHandle plan_backward;

	cufftPlan1d(&plan_forward, x.size(), CUFFT_R2C, 1);
	cufftPlan1d(&plan_backward, x.size(), CUFFT_C2R, 1);

	cufftReal* x_cuda = (cufftReal*)thrust::raw_pointer_cast(x.data());
	cuFloatComplex* fft_cuda = (cuFloatComplex*)thrust::raw_pointer_cast(fft.data());

	cufftExecR2C(plan_forward, x_cuda, fft_cuda);
	cufftExecC2R(plan_backward, fft_cuda, x_cuda);

	p_ = x;
}
