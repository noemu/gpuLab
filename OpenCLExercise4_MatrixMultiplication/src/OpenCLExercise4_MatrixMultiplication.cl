#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void matrixMulKernel1(__global float* h_inputA,__global float* h_inputB,__global float* h_outputC, size_t countAX_BY) {
	float sum = 0;
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	size_t countBX = get_global_size(0);
	size_t countAY = get_global_size(1);



	for (size_t k = 0; k < countAX_BY; k++) {
		float a = h_inputA[k + j * countAX_BY];
		float b = h_inputB[i + k * countBX];
		sum += a * b;
	}
	h_outputC[i + j * countBX] = sum;
}

// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

__attribute__((reqd_work_group_size(WG_SIZE, WG_SIZE, 1)))
__kernel void matrixMulKernel2(__global float* h_inputA,__global float* h_inputB,__global float* h_outputC, size_t countAX_BY) {

	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	size_t countBX = get_global_size(0);
	size_t countAY = get_global_size(1);

	//size_t WG_SIZE = get_local_size(0);

	__local float l_A[WG_SIZE ][ WG_SIZE ];
	__local float l_B[WG_SIZE ][ WG_SIZE ];

	float sum = 0;
	int kx = get_local_id(0);
	int ky = get_local_id(1);

	for (uint bs = 0; bs < countAX_BY; bs += WG_SIZE) {

		l_A [get_local_id(1)][get_local_id(0)] = h_inputA[(bs+kx) + j * countAX_BY ];
		l_B [get_local_id(1)][get_local_id(0)] = h_inputB[i + (bs+ky) * countBX ];

		barrier(CLK_LOCAL_MEM_FENCE);


		for (size_t k = 0; k < WG_SIZE; k++) {
			sum += l_A[get_local_id(1)][k] * l_B[k][get_local_id(0)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}



	h_outputC[i + j * countBX] = sum;

}
