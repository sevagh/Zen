#ifndef RHYTHM_TOOLKIT_H
#define RHYTHM_TOOLKIT_H

#include <stdexcept>

namespace rhythm_toolkit {
class RtkException : public std::runtime_error {
public:
	RtkException(std::string msg)
	    : std::runtime_error(msg){};
};
}; // namespace rhythm_toolkit

#endif /* RHYTHM_TOOLKIT_H */
