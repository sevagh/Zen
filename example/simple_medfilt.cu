#include <npp.h>
#include <iostream>
#include <thrust/device_vector.h>

void ex1() {
    int width  = 9;
    int height = 9;

    NppiSize  roi       = {width-2, height};
    NppiSize  mask      = {3, 1};
    NppiPoint anchor    = {1, 0};

    thrust::device_vector<float> _d_src(width*height);
    thrust::device_vector<float> _d_dst(width*height);

    Npp32f *d_src = (Npp32f*)thrust::raw_pointer_cast(_d_src.data());
    Npp32f *d_dst = (Npp32f*)thrust::raw_pointer_cast(_d_dst.data());

    Npp32s nSrcStep = height*sizeof(Npp32f);
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;
    Npp32u nBufferSize = 0;
    nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);

    nppiFilterMedian_32f_C1R(d_src + 1, nSrcStep, d_dst + 1, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
}

void ex2() {
    int width  = 9;
    int height = 9;

    NppiSize  roi       = {width, height-2};
    NppiSize  mask      = {1, 3};
    NppiPoint anchor    = {0, 1};

    thrust::device_vector<float> _d_src(width*height);
    thrust::device_vector<float> _d_dst(width*height);

    Npp32f *d_src = (Npp32f*)thrust::raw_pointer_cast(_d_src.data());
    Npp32f *d_dst = (Npp32f*)thrust::raw_pointer_cast(_d_dst.data());

    Npp32s nSrcStep = (height-2)*sizeof(Npp32f);
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;
    Npp32u nBufferSize = 0;
    nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);

    nppiFilterMedian_32f_C1R(d_src + nSrcStep, nSrcStep, d_dst + nSrcStep, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
}

int main(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();
    std::cout << "NPP Library Version: " << libVer->major << "." << libVer->minor << "." << libVer->build << std::endl;

    //ex1();
    ex2();

    return 0;
}
