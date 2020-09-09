#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include <clipp.h>
#include <offline.h>
#include <fakert.h>

int main(int argc, char* argv[])
{
    using namespace clipp;

    enum class mode { none, help, offline, realtime, fakert };
    mode selected = mode::none;

    std::string infile = "";
    std::string outfile = "";
    std::string indevice = "";
    std::string outdevice = "";

    zg::offline::OfflineParams offline_params;
    zg::fakert::FakeRtParams fakert_params;

    auto offline  = ( command("offline").set(selected,mode::offline),
		      required("-i", "--input") & value("infile", offline_params.infile)  % "input wav file",
		      (option("--hps").set(offline_params.do_hps,false) & opt_value("hop-h", offline_params.hop_h) & opt_value("beta-h", offline_params.beta_h) & opt_value("hop-p", offline_params.hop_p) & opt_value("beta-p", offline_params.beta_p)) % "2-pass HPR-iterative, defaults: harmonic=4096,2.0 percussive=256,2.0",
		      option("-o", "--output") & value("outfile", offline_params.outfile) % "(optional) output wav file"
		      ) % "offline (process entire songs at a time)";

    auto fakert  = ( command("fakert").set(selected,mode::fakert),
		      required("-i", "--input") & value("infile", fakert_params.infile)  % "input wav file",
		      (option("--hps").set(fakert_params.do_hps,false) & opt_value("hop", fakert_params.hop) & opt_value("beta", fakert_params.beta)) % "1-pass P-realtime, defaults: 256,2.5",
		      option("-o", "--output") & value("outfile", fakert_params.outfile) % "(optional) output wav file"
		      ) % "fakert (use slimmer rt algorithms with wav files)";

    auto zgcli = (
        offline | fakert
        | command("help", "-h", "--help").set(selected,mode::help)     % "Show this screen."
        | command("version", "-v", "--version")([]{ std::cout << "version 1.0\n"; }) % "Show version."
    );

    auto fmt = doc_formatting{}
        .first_column(2)
        .doc_column(20)
        .max_flags_per_param_in_usage(8);

    zg::offline::OfflineCommand zo(offline_params);
    zg::fakert::FakeRtCommand zf(fakert_params);

    if(parse(argc, argv, zgcli)) {
        switch(selected) {
            default:
            case mode::none:
                break;
            case mode::help: {
                std::cout << "zengarden\n\nUsage:\n"
                     << usage_lines(zgcli, "zengarden", fmt)
                     << "\n\nOptions:\n"
                     << documentation(zgcli, fmt) << '\n';
                }
                break;
            case mode::offline:
		return zo.execute();
            case mode::fakert:
		return zf.execute();
            case mode::realtime:
		std::cout << "realtime!" << std::endl;
                break;
        }
    }
    else {
        std::cerr << "Wrong command line arguments.\nUsage:\n"
                  << usage_lines(zgcli, "zengarden", fmt) << '\n';
    }
}