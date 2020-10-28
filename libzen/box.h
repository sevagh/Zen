#ifndef ZG_BOX_INTERNAL_H
#define ZG_BOX_INTERNAL_H

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

#include <libzen/zen.h>
#include <mfilt.h>

using namespace zen::internal::hps::mfilt;

namespace zen {
namespace internal {
	namespace hps {
		namespace box {
			class BoxFilterGPU {
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

				int smaller_nstep;
				int bigger_nstep;
				int smaller_start_pixel_offset;
				int bigger_start_pixel_offset;

				Npp32f* tmp_bigger_src;
				Npp32f* tmp_bigger_dst;

				// use time and frequency as axis names
				BoxFilterGPU(int time,
				             int frequency,
				             int filter_len,
				             MedianFilterDirection dir)
				    : mydir(dir)
				    , time(time)
				    , frequency(frequency)
				    , filter_len(filter_len)
				    , smaller_nstep(frequency
				                    * sizeof(Npp32f)) // expect 1D linear
				                                      // memory layout
				                                      // e.g. i*y + j
				    , smaller_roi(NppiSize{frequency, time})
				    , bigger_roi(NppiSize{frequency, time})
				    , roi(NppiSize{frequency, time})
				{
					if (((dir == MedianFilterDirection::TimeCausal
					      || dir == MedianFilterDirection::TimeAnticausal)
					     && filter_len > time)
					    || (dir == MedianFilterDirection::Frequency
					        && filter_len > frequency)) {
						throw zen::ZgException("box filter bigger than "
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
						// is the last frame, and should have the most box
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
						// mask, to get the best box filtering at the middle
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
						// choose an roi + mask + anchor so that the box filter
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
				};

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

					nppiFilterBoxBorder_32f_C1R(
					    tmp_bigger_src + bigger_start_pixel_offset,
					    bigger_nstep, roi, NppiPoint{0, 0},
					    tmp_bigger_dst + bigger_start_pixel_offset,
					    bigger_nstep, roi, mask, anchor, NPP_BORDER_REPLICATE);

					nppiCopy_32f_C1R(tmp_bigger_dst + bigger_start_pixel_offset,
					                 bigger_nstep, dst_ptr, smaller_nstep, roi);

					//nppiFilterBox_32f_C1R(
					//    src_ptr + smaller_start_pixel_offset, smaller_nstep,
					//    dst_ptr + smaller_start_pixel_offset, smaller_nstep,
					//    smaller_roi, mask, anchor);

					nppiFilterBoxBorder_32f_C1R(
					    src_ptr + smaller_start_pixel_offset, smaller_nstep,
					    roi, NppiPoint{0, 0},
					    dst_ptr + smaller_start_pixel_offset, smaller_nstep,
					    smaller_roi, mask, anchor, NPP_BORDER_REPLICATE);
				}
			};

			class BoxFilterCPU {
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
				BoxFilterCPU(int time,
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
						throw zen::ZgException("box filter bigger than "
						                       "matrix dimension");
					}

					filter_len += (1 - (filter_len % 2)); // make sure filter
					                                      // length is odd

					// roi selection and box filter parameters are much simpler
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

					IppStatus ipp_status = ippiFilterBoxBorderGetBufferSize(
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

				~BoxFilterCPU() { ippsFree(buffer); }

				void filter(std::vector<float>& src, std::vector<float>& dst)
				{
					ippiFilterBoxBorder_32f_C1R(
					    ( Ipp32f* )src.data(), nstep, ( Ipp32f* )dst.data(),
					    nstep, roi, mask, ippBorderRepl, 0, buffer);
				}
			};
		}; // namespace box
	};     // namespace hps
};         // namespace internal
};         // namespace zen

#endif /* ZG_BOX_INTERNAL_H */
