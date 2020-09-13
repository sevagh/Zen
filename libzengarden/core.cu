#include <core.h>
#include <npp.h>

static void zg_core_init() __attribute__((constructor));

void zg_core_init() { cudaSetDeviceFlags(cudaDeviceMapHost); }
