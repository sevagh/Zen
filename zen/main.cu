#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <clipp.h>
#include <fakert.h>
#include <offline.h>

int main(int argc, char* argv[])
{
	using namespace clipp;

	enum class mode { none, help, offline, fakert };
	mode selected = mode::none;

	zen::offline::OfflineParams offline_params{};
	zen::fakert::FakeRtParams fakert_params{};

	auto offline
	    = (command("offline").set(selected, mode::offline),
	       required("-i", "--input")
	           & value("infile", offline_params.infile) % "input wav file",
	       (option("--hps").set(offline_params.do_hps, true)
	        & opt_value("hop-h", offline_params.hop_h)
	        & opt_value("beta-h", offline_params.beta_h)
	        & opt_value("hop-p", offline_params.hop_p)
	        & opt_value("beta-p", offline_params.beta_p))
	           % "2-pass HPR-iterative, defaults: harmonic=4096,2.0 "
	             "percussive=256,2.0",
	       option("-o", "--out-prefix")
	           & value("outfile_prefix", offline_params.outfile_prefix)
	                 % "(optional) output wav file prefix",
	       option("--cpu").set(offline_params.cpu, true),
	       option("--nocopybord").set(offline_params.nocopybord, true))
	      % "offline (process entire songs at a time)";

	auto fakert
	    = (command("fakert").set(selected, mode::fakert),
	       required("-i", "--input")
	           & value("infile", fakert_params.infile) % "input wav file",
	       (option("--hps").set(fakert_params.do_hps, true)
	        & opt_value("hop", fakert_params.hop)
	        & opt_value("beta", fakert_params.beta))
	           % "1-pass P-realtime, defaults: 256,2.5",
	       option("-o", "--output")
	           & value("outfile", fakert_params.outfile)
	                 % "(optional) output wav file",
	       option("--cpu").set(fakert_params.cpu, true),
	       option("--nocopybord").set(fakert_params.nocopybord, true))
	      % "fakert (use slim rt algorithms with wav files)";

	auto zencli = (offline | fakert
	              | command("help", "-h", "--help").set(selected, mode::help)
	                    % "Show this screen."
	              | command("version", "-v", "--version")([] {
		                std::cout << "version 1.0\n";
	                }) % "Show version.");

	auto fmt = doc_formatting{}
	               .first_column(2)
	               .doc_column(20)
	               .max_flags_per_param_in_usage(8);

	if (parse(argc, argv, zencli)) {
		switch (selected) {
		default:
		case mode::none:
			break;
		case mode::help: {
			std::cout << "usage:\n\n"
			          << usage_lines(zencli, "zen", fmt)
			          << "\n\nOptions:\n"
			          << documentation(zencli, fmt) << '\n';
		} break;
		case mode::offline: {
			zen::offline::OfflineCommand zo(offline_params);
			return zo.execute();
		}
		case mode::fakert: {
			zen::fakert::FakeRtCommand zf(fakert_params);
			return zf.execute();
		}
		}
	}
	else {
		std::cerr << "usage:\n\n"
		          << usage_lines(zencli, "zen", fmt) << '\n';
	}
}
