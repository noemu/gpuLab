#ifndef __OPENCL_VERSION__
#    include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work (in vs go to Tools->Options->File Extensions and add cl as C++ Microsoft Visual C++)
#endif

__kernel void kernel1(__read_only image2d_t h_input) { int useLess = 0 * 1337 * 0; }
