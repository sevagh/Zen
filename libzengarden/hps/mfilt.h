#ifndef ZG_MFILT_INTERNAL_H
#define ZG_MFILT_INTERNAL_H

#include <complex>
#include <cstddef>
#include <vector>

#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <npp.h>
#include <nppdefs.h>
#include <nppi.h>

#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>

#include <libzengarden/zg.h>

namespace zg {
namespace internal {
namespace hps {
	namespace mfilt {
		enum MedianFilterDirection {
			TimeCausal,
			TimeAnticausal,
			Frequency,
		};

		class MedianFilterGPU {
		public:
			MedianFilterDirection mydir;

			int time;
			int frequency;
			int filter_len;

			NppiSize smaller_roi;
			NppiSize bigger_roi;
			NppiSize roi;

			int filter_mid;
			NppiSize mask;
			NppiPoint anchor;
			Npp8u* buffer;
			Npp32u buffer_size;

			int smaller_nstep;
			int bigger_nstep;
			int smaller_start_pixel_offset;
			int bigger_start_pixel_offset;

			bool copy_bord;
			Npp32f* tmp_bigger_src;
			Npp32f* tmp_bigger_dst;

			// use time and frequency as axis names
			MedianFilterGPU(int time,
			                int frequency,
			                int filter_len,
			                MedianFilterDirection dir,
			                bool copy_bord
			                = false) // copy borders - better results, slower
			    : mydir(dir)
			    , time(time)
			    , frequency(frequency)
			    , filter_len(filter_len)
			    , smaller_nstep(frequency * sizeof(Npp32f)) // expect 1D linear
			                                                // memory layout
			                                                // e.g. i*y + j
			    , smaller_roi(NppiSize{frequency, time})
			    , bigger_roi(NppiSize{frequency, time})
			    , roi(NppiSize{frequency, time})
			    , copy_bord(copy_bord)
			{
				if (((dir == MedianFilterDirection::TimeCausal
				      || dir == MedianFilterDirection::TimeAnticausal)
				     && filter_len > time)
				    || (dir == MedianFilterDirection::Frequency
				        && filter_len > frequency)) {
					throw zg::ZgException("median filter bigger than "
					                      "matrix dimension");
				}

				filter_len += (1 - (filter_len % 2)); // make sure filter
				                                      // length is odd
				filter_mid = ( int )floorf(( float )filter_len / 2);

				switch (dir) {
				// https://docs.nvidia.com/cuda/npp/nppi_conventions_lb.html#roi_specification
				case MedianFilterDirection::TimeCausal:
					// causal case is for real-time use where future frames
					// aren't available
					//
					// in this case the sliding stft should have a trailing
					// history of l_harm/2 frames (going by the original
					// offline algorithm)
					//
					// we're supplying a full matrix of time = stft_width =
					// l_harm
					//
					// our mask should extend backwards, i.e. have the anchor
					// at the tip of the mask this is because the current frame
					// is the last frame, and should have the most median
					// filtering - the earlier frames, or past, lose importance
					// rapidly
					smaller_roi.height -= filter_len;
					bigger_roi.height += filter_len;

					mask = NppiSize{1, filter_len};
					anchor = NppiPoint{0, filter_len};

					// start one entire mask past the beginning so the mask can
					// validly extend backwards
					smaller_start_pixel_offset
					    = filter_len * smaller_nstep / sizeof(Npp32f);

					break;
				case MedianFilterDirection::TimeAnticausal:
					// anticausal case is for offline processing where we can
					// use past and future frames for improved
					// harmonic-percussive separation
					//
					// in this case we expect to be supplied with time =
					// stft_width = l_harm but we have past and future samples
					//
					// therefore the anchor should be in the middle of the
					// mask, to get the best median filtering at the middle
					// frame which contains the audio to be reconstructed
					smaller_roi.height -= filter_len;
					bigger_roi.height += filter_len;

					mask = NppiSize{1, filter_len};
					anchor = NppiPoint{0, filter_mid};

					// start half a mask past the beginning so the mask can
					// validly extend backwards
					smaller_start_pixel_offset
					    = filter_mid * smaller_nstep / sizeof(Npp32f);

					break;
				case MedianFilterDirection::Frequency:
					// choose an roi + mask + anchor so that the median filter
					// extends forward in the fft
					//
					// since we're discarding the second half of the fft, it's
					// less important
					smaller_roi.width -= filter_len;
					bigger_roi.width += filter_len;

					mask = NppiSize{filter_len, 1};
					anchor = NppiPoint{0, 0};

					smaller_start_pixel_offset = 0;

					break;
				}

				if (copy_bord) {
					tmp_bigger_src = nppiMalloc_32f_C1(
					    bigger_roi.width, bigger_roi.height, &bigger_nstep);
					if (tmp_bigger_src == 0) {
						std::cerr << "nppiMalloc error" << std::endl;
						std::exit(1);
					}
					tmp_bigger_dst = nppiMalloc_32f_C1(
					    bigger_roi.width, bigger_roi.height, &bigger_nstep);
					if (tmp_bigger_dst == 0) {
						std::cerr << "nppiMalloc error" << std::endl;
						std::exit(1);
					}

					// adjust starting offset for padded/border-copied bigger
					// line size
					if (dir == MedianFilterDirection::Frequency) {
						bigger_start_pixel_offset = 0;
					}
					else if (dir == MedianFilterDirection::TimeCausal) {
						bigger_start_pixel_offset
						    = filter_len * bigger_nstep / sizeof(Npp32f);
					}
					else if (dir == MedianFilterDirection::TimeAnticausal) {
						bigger_start_pixel_offset
						    = filter_mid * bigger_nstep / sizeof(Npp32f);
					}

					NppStatus npp_status
					    = nppiFilterMedianGetBufferSize_32f_C1R(
					        roi, mask, &buffer_size);
					if (npp_status != NPP_NO_ERROR) {
						std::cerr << "NPP error " << npp_status << std::endl;
						std::exit(1);
					}
				}
				else {
					NppStatus npp_status
					    = nppiFilterMedianGetBufferSize_32f_C1R(
					        smaller_roi, mask, &buffer_size);
					if (npp_status != NPP_NO_ERROR) {
						std::cerr << "NPP error " << npp_status << std::endl;
						std::exit(1);
					}
				}
				cudaError cuda_status
				    = cudaMalloc(( void** )&buffer, buffer_size);
				if (cuda_status != cudaSuccess) {
					std::cerr << cudaGetErrorString(cuda_status);
					std::exit(1);
				}
			};

			~MedianFilterGPU()
			{
				cudaFree(buffer);
				if (copy_bord) {
					nppsFree(tmp_bigger_src);
					nppsFree(tmp_bigger_dst);
				}
			}

			void filter(thrust::device_vector<float>& src,
			            thrust::device_vector<float>& dst)
			{
				filter(src.data(), dst.data());
			}

			void filter(thrust::device_ptr<float> src,
			            thrust::device_ptr<float> dst)
			{
				auto src_ptr = ( Npp32f* )thrust::raw_pointer_cast(src);
				auto dst_ptr = ( Npp32f* )thrust::raw_pointer_cast(dst);

				if (!copy_bord) {
					nppiFilterMedian_32f_C1R(
					    src_ptr + smaller_start_pixel_offset, smaller_nstep,
					    dst_ptr + smaller_start_pixel_offset, smaller_nstep,
					    smaller_roi, mask, anchor, buffer);
				}
				else {
					if (mydir == MedianFilterDirection::Frequency) {
						nppiCopyWrapBorder_32f_C1R(
						    src_ptr, smaller_nstep, roi, tmp_bigger_src,
						    bigger_nstep, bigger_roi, 0, filter_mid);
					}
					else {
						nppiCopyWrapBorder_32f_C1R(
						    src_ptr, smaller_nstep, roi, tmp_bigger_src,
						    bigger_nstep, bigger_roi, filter_mid, 0);
					}

					nppiFilterMedian_32f_C1R(
					    tmp_bigger_src + bigger_start_pixel_offset,
					    bigger_nstep,
					    tmp_bigger_dst + bigger_start_pixel_offset,
					    bigger_nstep, roi, mask, anchor, buffer);

					nppiCopy_32f_C1R(tmp_bigger_dst + bigger_start_pixel_offset,
					                 bigger_nstep, dst_ptr, smaller_nstep, roi);
				}
			}
		};

		class MedianFilterCPU {
		public:
			int time;
			int frequency;
			int filter_len;

			IppiSize roi;
			IppiSize mask;

			int nstep;

			Ipp8u* buffer;
			int buffer_size;

			// use time and frequency as axis names
			MedianFilterCPU(int time,
			                int frequency,
			                int filter_len,
			                MedianFilterDirection dir)
			    : time(time)
			    , frequency(frequency)
			    , filter_len(filter_len)
			    , nstep(frequency * sizeof(Ipp32f))
			    , roi(IppiSize{frequency, time})
			{
				if (((dir == MedianFilterDirection::TimeCausal
				      || dir == MedianFilterDirection::TimeAnticausal)
				     && filter_len > time)
				    || (dir == MedianFilterDirection::Frequency
				        && filter_len > frequency)) {
					throw zg::ZgException("median filter bigger than "
					                      "matrix dimension");
				}

				filter_len += (1 - (filter_len % 2)); // make sure filter
				                                      // length is odd

				// roi selection and median filter parameters are much simpler
				// for ipp since it handles border replication
				switch (dir) {
				case MedianFilterDirection::TimeCausal:
				case MedianFilterDirection::TimeAnticausal:
					mask = IppiSize{1, filter_len};
					break;
				case MedianFilterDirection::Frequency:
					mask = IppiSize{filter_len, 1};
					break;
				}

				IppStatus ipp_status = ippiFilterMedianBorderGetBufferSize(
				    roi, mask, ipp32f, 1, &buffer_size);
				if (ipp_status < 0) {
					std::cerr << "IPP error " << ipp_status << std::endl;
					std::exit(1);
				}

				buffer = ippsMalloc_8u(buffer_size);
				if (buffer == nullptr) {
					std::cerr << "IPP malloc error " << std::endl;
					std::exit(1);
				}
			};

			~MedianFilterCPU() { ippsFree(buffer); }

			void filter(std::vector<float>& src, std::vector<float>& dst)
			{
				ippiFilterMedianBorder_32f_C1R(
				    ( Ipp32f* )src.data(), nstep, ( Ipp32f* )dst.data(), nstep,
				    roi, mask, ippBorderRepl, 0, buffer);
			}
		};
	}; // namespace mfilt
};     // namespace hps
};     // namespace internal
};     // namespace zg

#endif /* ZG_MFILT_INTERNAL_H */
