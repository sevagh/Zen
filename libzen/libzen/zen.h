#ifndef ZG_PUB_H
#define ZG_PUB_H

#include <stdexcept>

namespace zen {

class ZgException : public std::runtime_error {
public:
	ZgException(std::string msg)
	    : std::runtime_error(msg){};
};

enum Backend { GPU, CPU };

constexpr float Eps = std::numeric_limits<float>::epsilon();

}; // namespace zen

#endif /* ZG_PUB_H */
