#include <npp.h>
#include <iostream>
#include <thrust/device_vector.h>

void ex1() {
    int width  = 9;
    int height = 9;
    int masklen = 5;

    int maskmid = masklen/2;

    NppiSize  roi       = {width-masklen, height};
    NppiSize  mask      = {masklen, 1};
    NppiPoint anchor    = {maskmid, 0};

    Npp32u nBufferSize = 0;
    NppStatus status   = NPP_SUCCESS;

    thrust::device_vector<float> _d_src(width*height);
    thrust::device_vector<float> _d_dst(width*height);

    Npp32f *d_src = (Npp32f*)thrust::raw_pointer_cast(_d_src.data());
    Npp32f *d_dst = (Npp32f*)thrust::raw_pointer_cast(_d_dst.data());

    Npp32s nSrcStep = sizeof(Npp32f) * width;
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;

    status = nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);

    status = nppiFilterMedian_32f_C1R(d_src + maskmid, nSrcStep, d_dst + maskmid, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
}

void ex2() {
    int width  = 9;
    int height = 9;
    int masklen = 5;

    int maskmid = masklen/2;

    NppiSize  roi       = {width-masklen, height};
    NppiSize  mask      = {1, masklen};
    NppiPoint anchor    = {0, maskmid};

    Npp32u nBufferSize = 0;
    NppStatus status   = NPP_SUCCESS;

    thrust::device_vector<float> _d_src(width*height);
    thrust::device_vector<float> _d_dst(width*height);

    Npp32f *d_src = (Npp32f*)thrust::raw_pointer_cast(_d_src.data());
    Npp32f *d_dst = (Npp32f*)thrust::raw_pointer_cast(_d_dst.data());

    Npp32s nSrcStep = sizeof(Npp32f) * width;
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;

    status = nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);

    status = nppiFilterMedian_32f_C1R(d_src + maskmid*nSrcStep, nSrcStep, d_dst + maskmid*nSrcStep, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
}

int main(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();
    std::cout << "NPP Library Version: " << libVer->major << "." << libVer->minor << "." << libVer->build << std::endl;

    ex1();
    //ex2();

    return 0;
}
