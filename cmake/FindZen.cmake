find_path(Zen_INCLUDE_DIR NAMES libzen/zen.h)

find_library(Zen_LIBRARY NAMES libzen)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Zen DEFAULT_MSG Zen_LIBRARY Zen_INCLUDE_DIR)

mark_as_advanced(Zen_INCLUDE_DIR Zen_LIBRARY)
