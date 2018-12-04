#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void prefixSumKernel(__global const int* d_input, __global int* d_output, __global int* d_sum) {

	__local float temp[WG_SIZE];

	int itemId = get_global_id(0);
	int workId = get_local_id(0);
	int workGroupNumber = get_group_id(0);

	temp[workId] = d_input[itemId];
	barrier(CLK_LOCAL_MEM_FENCE);

	//int tempVal = 0;

	for(int d = 1; d < WG_SIZE; d*=2){
		if(workId - d >= 0.0f){
			int tmp = temp[workId-d];
			barrier(CLK_LOCAL_MEM_FENCE);
			temp[workId] = temp[workId]+tmp;
		}else{
			//break;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//temp[workId] = tempVal;

	if(workId ==  WG_SIZE-1){
		d_sum[workGroupNumber] = temp[workId];
	}

	d_output[itemId] = temp[workId];

}

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void blockAddKernel(__global const int* d_input,__global const int* s_sum, __global int* d_output) {

	int offset = 0;
	int group_id = get_group_id(0);

	if(group_id != 0) offset = s_sum[group_id-1];

	d_output[get_global_id(0)] = offset + d_input[get_global_id(0)];

}