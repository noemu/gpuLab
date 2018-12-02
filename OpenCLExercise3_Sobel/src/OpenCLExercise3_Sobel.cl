#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif



float getValueGlobal(__read_only image2d_t image,int countX,int countY,int i, int j) {
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	return read_imagef(image, sampler, (int2){i, j}).x;
}

__kernel void sobelKernel1(__read_only image2d_t h_input, int countX, int countY,__global float* d_output) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+
			2*getValueGlobal(h_input, countX, countY, i-1, j)+
			getValueGlobal(h_input, countX, countY, i-1, j+1)-
			getValueGlobal(h_input, countX, countY, i+1, j-1)-
			2*getValueGlobal(h_input, countX, countY, i+1, j)-
			getValueGlobal(h_input, countX, countY, i+1, j+1);

	float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+
			2*getValueGlobal(h_input, countX, countY, i, j-1)+
			getValueGlobal(h_input, countX, countY, i+1, j-1)-
			getValueGlobal(h_input, countX, countY, i-1, j+1)-
			2*getValueGlobal(h_input, countX, countY, i, j+1)-
			getValueGlobal(h_input, countX, countY, i+1, j+1);

	d_output[countX * j + i] = sqrt(Gx * Gx + Gy * Gy);

}

__kernel void sobelKernel2(__read_only image2d_t h_input, int countX, int countY,__global float* d_output) {

	int i = get_global_id(0);
	int j = get_global_id(1);


	float mm = getValueGlobal(h_input, countX, countY, i-1, j-1);
	float mp = getValueGlobal(h_input, countX, countY, i-1, j+1);
	float pm = getValueGlobal(h_input, countX, countY, i+1, j-1);
	float pp = getValueGlobal(h_input, countX, countY, i+1, j+1);

	float Gx = mm+
			2*getValueGlobal(h_input, countX, countY, i-1, j)+
			mp-
			pm-
			2*getValueGlobal(h_input, countX, countY, i+1, j)-
			pp;

	float Gy = mm+
			2*getValueGlobal(h_input, countX, countY, i, j-1)+
			pm-
			mp-
			2*getValueGlobal(h_input, countX, countY, i, j+1)-
			pp;

	d_output[countX * j + i] = sqrt(Gx * Gx + Gy * Gy);

}

__kernel void sobelKernel3(__read_only image2d_t h_input, int countX, int countY,__global float* d_output) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+
			2*getValueGlobal(h_input, countX, countY, i-1, j)+
			getValueGlobal(h_input, countX, countY, i-1, j+1)-
			getValueGlobal(h_input, countX, countY, i+1, j-1)-
			2*getValueGlobal(h_input, countX, countY, i+1, j)-
			getValueGlobal(h_input, countX, countY, i+1, j+1);

	float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+
			2*getValueGlobal(h_input, countX, countY, i, j-1)+
			getValueGlobal(h_input, countX, countY, i+1, j-1)-
			getValueGlobal(h_input, countX, countY, i-1, j+1)-
			2*getValueGlobal(h_input, countX, countY, i, j+1)-
			getValueGlobal(h_input, countX, countY, i+1, j+1);

	d_output[countX * j + i] = sqrt(Gx * Gx + Gy * Gy);

}
