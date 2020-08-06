find_path(FFTS_INCLUDE_DIR NAMES ffts/ffts.h)

find_library(FFTS_LIBRARY NAMES ffts)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTS DEFAULT_MSG FFTS_LIBRARY FFTS_INCLUDE_DIR)

mark_as_advanced(FFTS_INCLUDE_DIR FFTS_LIBRARY)
