#include <npp.h>
#include <iostream>

void ex1() {
    unsigned int width  = 1024;
    unsigned int height = 440;
    NppiSize  roi       = {width, height};
    NppiSize  mask      = {3, 1};
    NppiPoint anchor    = {0, 0};

    Npp32u nBufferSize = 0;
    NppStatus status   = NPP_SUCCESS;

    Npp16u *d_src = NULL;
    Npp16u *d_dst = NULL;

    cudaMalloc((void **)(&d_src), sizeof(Npp16u) * width * height);
    cudaMalloc((void **)(&d_dst), sizeof(Npp16u) * width * height);

    Npp32s nSrcStep = sizeof(Npp16u) * width;
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;
    status = nppiFilterMedianGetBufferSize_16u_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);
    status = nppiFilterMedian_16u_C1R(d_src, nSrcStep, d_dst, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void ex2() {
    unsigned int width  = 17;
    unsigned int height = 8;
    NppiSize  roi       = {width, height};
    NppiSize  mask      = {1, 3};
    NppiPoint anchor    = {0, 0};

    Npp32u nBufferSize = 0;
    NppStatus status   = NPP_SUCCESS;

    Npp16u *d_src = NULL;
    Npp16u *d_dst = NULL;

    cudaMalloc((void **)(&d_src), sizeof(Npp16u) * width * height);
    cudaMalloc((void **)(&d_dst), sizeof(Npp16u) * width * height);

    Npp32s nSrcStep = sizeof(Npp16u) * height;
    Npp32s nDstStep = nSrcStep;

    Npp8u *d_median_filter_buffer = NULL;
    status = nppiFilterMedianGetBufferSize_16u_C1R(roi, mask, &nBufferSize);
    cudaMalloc((void **)(&d_median_filter_buffer), nBufferSize);
    status = nppiFilterMedian_16u_C1R(d_src, nSrcStep, d_dst, nDstStep, roi, mask, anchor, d_median_filter_buffer);

    cudaFree(d_median_filter_buffer);
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();
    std::cout << "NPP Library Version: " << libVer->major << "." << libVer->minor << "." << libVer->build << std::endl;

    ex1();
    ex2();

    return 0;
}
