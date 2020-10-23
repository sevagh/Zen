#ifndef ZG_HPS_PUB_H
#define ZG_HPS_PUB_H

#include <array>
#include <complex>
#include <cstddef>
#include <thrust/device_vector.h>
#include <vector>

#include <libzen/io.h>
#include <libzen/zen.h>

// forward declare private implementations
namespace zen {
namespace internal {
	namespace hps {
		template <zen::Backend B>
		class HPR;
	}; // namespace hps
};     // namespace internal
};     // namespace zen

namespace zen {
namespace hps {
	template <zen::Backend B>
	class HPRIOffline {
	public:
		HPRIOffline(float fs,
		            std::size_t hop_h,
		            std::size_t hop_p,
		            float beta_h,
		            float beta_p);

		// set nocopybord=true to disable NPP's internal copyborder for median filter
		// gain some performance, lose some quality
		HPRIOffline(float fs,
		            std::size_t hop_h,
		            std::size_t hop_p,
		            float beta_h,
		            float beta_p,
		            bool nocopybord);

		HPRIOffline(float fs, std::size_t hop_h, std::size_t hop_p);

		HPRIOffline(float fs);
		~HPRIOffline();

		// pass the entire song in the in vec
		// the vector is copied to be modified in-place
		//
		// returns a triplet of harmonic,percussive,residual results
		// of audio.size()
		std::array<std::vector<float>, 3> process(std::vector<float> audio);

		void use_sse_filter();
		void use_soft_mask();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		// we use 2 cascading HPR objects to implement driedger's
		// offline iterative algorithm "HPR-I"
		zen::internal::hps::HPR<B>* p_impl_h;
		zen::internal::hps::HPR<B>* p_impl_p;

		std::size_t hop_h, hop_p;
		zen::io::IOGPU io_h;
		zen::io::IOGPU io_p;
	};

	template <zen::Backend B>
	class PRealtime {
	public:
		PRealtime(float fs, std::size_t hop, float beta);

		// set nocopybord=true to disable NPP's internal copyborder for median filter
		// gain some performance, lose some quality
		PRealtime(float fs, std::size_t hop, float beta, bool nocopybord);
		PRealtime(float fs, std::size_t hop);
		PRealtime(float fs);
		~PRealtime();

		// copies from the io in vec, writes to the io out vec
		// pass in a real-time stream of the input, one hop at a time
		void process_next_hop(thrust::device_ptr<float> in,
		                      thrust::device_ptr<float> out);
		void process_next_hop(float* in, float* out);

		// gpu benefits from some warmup, especially in real-time contexts
		// where the slow early iterations cause latency issues
		void warmup();
		void warmup(zen::io::IOGPU& io);

		void use_sse_filter();
		void use_soft_mask();

	private:
		// https://en.cppreference.com/w/cpp/language/pimpl
		zen::internal::hps::HPR<B>* p_impl;
	};
}; // namespace hps
}; // namespace zen

#endif /* ZG_HPS_PUB_H */
