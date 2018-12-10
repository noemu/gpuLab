/**
	Canny-Edge Detector
	main.cpp
	Create and execute the GPU-Kernel.

	//TODO Surnames @author Ellen ..., Marvin ..., Phong ..., Rafael Jarosch
*/

#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

/**
	Create a context

	@return cl::Context
*/
cl::Context createContext() {

	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);
	return context;
}

/**
	Build-Kernel

	@param cl:Context
	@param deviceNr
	@return cl:program
*/
cl::Program buildKernel(cl::Context context, int deviceNr) {
	// Get a device of the context
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Declare some values
	std::size_t wgSize = 128;


	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/kernel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	// This will pass the value of wgSize as a preprocessor constant "WG_SIZE" to the OpenCL C compiler
	OpenCL::buildProgram(program, devices, "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));

	return program;
}

/**
	Main-Method

	@param argc: start-argument
	@return Error-Code
*/
int main(int argc, char** argv) {
	cl::Context context = createContext();

	cl::Program program = buildKernel(context, argc < 2 ? 1 : atoi(argv[1])); //if no start argument is given, use first device

	// Create a kernel
	cl::Kernel kernel(program, "kernel1");


	return 0;
}
