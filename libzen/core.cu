#include <core.h>
#include <npp.h>

static void zen_core_init() __attribute__((constructor));

void zen_core_init() { cudaSetDeviceFlags(cudaDeviceMapHost); }
