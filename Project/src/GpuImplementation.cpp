#include "GpuImplementation.h"

#include <Core/Assert.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

/**
        Create a context

        @return cl::Context
*/
cl::Context GpuImplementation::createContext() {
    // cl::Context context(CL_DEVICE_TYPE_GPU);
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
    cl_context_properties prop[4] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0};
    std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '"
              << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
    cl::Context context(CL_DEVICE_TYPE_GPU, prop);
    return context;
}

/**
        Build-Kernel

        @param cl:Context
        @param deviceNr
        @return cl:program
*/
cl::Program GpuImplementation::buildKernel(cl::Context context, int deviceNr) {
    // Get a device of the context
    std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT(deviceNr > 0);
    ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);

    // Create a command queue
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Declare some values
    std::size_t wgSize = 128;

    // Load the source code
    program = OpenCL::loadProgramSource(context, "src/kernel.cl");
    // Compile the source code. This is similar to program.build(devices) but will
    // print more detailed error messages This will pass the value of wgSize as a
    // preprocessor constant "WG_SIZE" to the OpenCL C compiler
    OpenCL::buildProgram(program, devices, "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));

    return program;
}

GpuImplementation::GpuImplementation(int deviceNr) {
    context = createContext();

    program = buildKernel(context, deviceNr); // if no start argument is given, use first device

    // Create a kernel
    kernel = cl::Kernel(program, "kernel1");
}

GpuImplementation::~GpuImplementation() {}

void GpuImplementation::execute() {
    std::size_t wgSizeX = 10; // Hard coded Work Group Size, should be
    std::size_t wgSizeY = 10;

    ASSERT(imageHeight % wgSizeY == 0); // imagageWidth/height shpould be dividable by wgSize
    ASSERT(imageWidth % wgSizeX == 0);

    kernel.setArg<cl::Image2D>(0, image);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight),
        cl::NDRange(wgSizeX, wgSizeY), NULL, &executionEvent);

    // copy to output
    int count = imageWidth * imageHeight;
    std::vector<float> h_outputGpu(count);
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = imageWidth;
    region[1] = imageHeight;
    region[2] = 1;
    queue.enqueueReadImage(
        image, true, origin, region, imageWidth * sizeof(float), 0, h_outputGpu.data(), NULL, &copyToHostEvent);
}

void GpuImplementation::printTimeMeasurement() {

    Core::TimeSpan gpuExecutionTime = OpenCL::getElapsedTime(executionEvent);
    Core::TimeSpan copyToClienTime = OpenCL::getElapsedTime(copyToClientEvent);
    Core::TimeSpan copyToHostTime = OpenCL::getElapsedTime(copyToHostEvent);

    std::stringstream outputString;

    const int firstColumnWidth = 24;
    const int generalColumnWidth = 19;

    outputString << std::left << std::setw(firstColumnWidth) << "GPU-Performance" << std::setw(generalColumnWidth)
                 << "copy to client" << std::setw(generalColumnWidth) << "copy to host" << std::setw(generalColumnWidth)
                 << "execution" << std::endl;
    outputString << std::left << std::setw(firstColumnWidth) << "" << std::setw(generalColumnWidth)
                 << copyToClienTime.getSeconds() << std::setw(generalColumnWidth) << copyToHostTime.getSeconds()
                 << std::setw(generalColumnWidth) << gpuExecutionTime.getSeconds() << std::endl;

    std::cout << outputString.str() << std::endl;
}

void GpuImplementation::loadImage(const boost::filesystem::path& filename) {

    // random init, for testing
    imageWidth = 200;
    imageHeight = 200;

    int count = imageWidth * imageHeight;
    std::vector<float> h_input(count);

    for (int i = 0; i < count; i++) h_input[i] = (rand() % 100) / 5.0f - 10.0f;

    // copyToClient
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = imageWidth;
    region[1] = imageHeight;
    region[2] = 1;

    image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);

    queue.enqueueWriteImage(
        image, true, origin, region, imageWidth * sizeof(float), 0, &(h_input[0]), NULL, &copyToClientEvent);
}