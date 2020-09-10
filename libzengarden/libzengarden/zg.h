#ifndef ZG_PUB_H
#define ZG_PUB_H

#include <stdexcept>

namespace zg {

class ZgException : public std::runtime_error {
public:
	ZgException(std::string msg)
	    : std::runtime_error(msg){};
};

enum Backend { GPU, CPU };
}; // namespace zg

#endif /* ZG_PUB_H */
