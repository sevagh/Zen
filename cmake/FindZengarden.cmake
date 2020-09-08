find_path(Zengarden_INCLUDE_DIR NAMES libzengarden/zg.h)

if(Zengarden_STATIC)
	find_library(Zengarden_LIBRARY NAMES libzengarden_static)
else()
	find_library(Zengarden_LIBRARY NAMES libzengarden)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Zengarden DEFAULT_MSG Zengarden_LIBRARY Zengarden_INCLUDE_DIR)

mark_as_advanced(Zengarden_INCLUDE_DIR Zengarden_LIBRARY)
