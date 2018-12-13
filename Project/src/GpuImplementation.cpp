#include "GpuImplementation.h"

#include <Core/Assert.hpp>
#include <Core/Image.hpp>

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

    // Load the source code
    program = OpenCL::loadProgramSource(context, "src/kernel.cl");
    // Compile the source code. This is similar to program.build(devices) but will
    // print more detailed error messages This will pass the value of wgSize as a
    // preprocessor constant "WG_SIZE" to the OpenCL C compiler
    OpenCL::buildProgram(program, devices,
        "-D WG_SIZE_X=" + boost::lexical_cast<std::string>(wgSizeX) +
            " -D WG_SIZE_Y=" + boost::lexical_cast<std::string>(wgSizeY));


    return program;
}

/**
    constructor

    @param deviceNr. The number of the device to be used
*/
GpuImplementation::GpuImplementation(int deviceNr) {
    context = createContext();

    program = buildKernel(context, deviceNr); // if no start argument is given, use first device

    // Create a kernel
    sobel_Kernel = cl::Kernel(program, "sobel1");
    gaussC_kernel = cl::Kernel(program, "gaussConvolution");
    nonMaxUp_Kernel = cl::Kernel(program, "nonMaximumSuppression");
    hysterese_kernel = cl::Kernel(program, "hysterese");
}

/**
    destructor
*/
GpuImplementation::~GpuImplementation() {}


/**
    executes the Kernel
*/
void GpuImplementation::execute(float T1 = 0.1, float T2 = 0.7)
{

    int count = imageWidth * imageHeight;
    std::vector<float> h_outputGpu(count);
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = imageWidth;
    region[1] = imageHeight;
    region[2] = 1;
    region[2] = 1;

    ASSERT(imageHeight % wgSizeY == 0); // imagageWidth/height should be dividable by wgSize
    ASSERT(imageWidth % wgSizeX == 0);


    // cl::Image2D full_strength(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);
    // //fix read_imagef from read_write
    cl::Image2D strength(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);
    cl::Image2D direction(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);
    cl::Image2D maximised_strength(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);
    cl::Buffer canny_Edge(context, CL_MEM_READ_WRITE, sizeof(float) * count);

    //
    // Gauss Smoothing Kernel
    //
    gaussC_kernel.setArg<cl::Image2D>(0, image);
    gaussC_kernel.setArg<cl::Image2D>(1, image);


    int countSmoothnessRuns = 1;
    for (int count = 0; count < countSmoothnessRuns; count++) {
        queue.enqueueNDRangeKernel(gaussC_kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight),
            cl::NDRange(wgSizeX, wgSizeY), NULL, &executionEvent);
    }

    // copy output to image
    queue.enqueueReadImage(
        image, true, origin, region, imageWidth * sizeof(float), 0, h_outputGpu.data(), NULL, &copyToHostEvent);
    Core::writeImagePGM("output_GaussSmoothed.pgm", h_outputGpu, imageWidth, imageHeight);


    //
    // Sobel Gradient Kernel
    //

    sobel_Kernel.setArg<cl::Image2D>(0, image);
    sobel_Kernel.setArg<cl::Image2D>(1, strength);
    sobel_Kernel.setArg<cl::Image2D>(2, direction);

    queue.enqueueNDRangeKernel(sobel_Kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight),
        cl::NDRange(wgSizeX, wgSizeY), NULL, &executionEvent);

    // copy output to image
    queue.enqueueReadImage(
        strength, true, origin, region, imageWidth * sizeof(float), 0, h_outputGpu.data(), NULL, &copyToHostEvent);
    Core::writeImagePGM("output_Gradient.pgm", h_outputGpu, imageWidth, imageHeight);

    //
    // Non Maximum Suppression
    //

    nonMaxUp_Kernel.setArg<cl::Image2D>(0, strength);
    nonMaxUp_Kernel.setArg<cl::Image2D>(1, direction);
    nonMaxUp_Kernel.setArg<cl::Image2D>(2, maximised_strength);

    queue.enqueueNDRangeKernel(nonMaxUp_Kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight),
        cl::NDRange(wgSizeX, wgSizeY), NULL, &executionEvent);

    // copy output to image
    queue.enqueueReadImage(maximised_strength, true, origin, region, imageWidth * sizeof(float), 0, h_outputGpu.data(),
        NULL, &copyToHostEvent);
    Core::writeImagePGM("output_NonMaxSup.pgm", h_outputGpu, imageWidth, imageHeight);

    //
    // Hysterese Kernel
    //

    hysterese_kernel.setArg<cl::Image2D>(0, maximised_strength);
    hysterese_kernel.setArg<cl::Buffer>(1, canny_Edge);
    hysterese_kernel.setArg<float>(2, T1);
    hysterese_kernel.setArg<float>(3, T2);

    queue.enqueueNDRangeKernel(hysterese_kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight),
        cl::NDRange(wgSizeX, wgSizeY), NULL, &executionEvent);

    // copy output to image
    queue.enqueueReadBuffer(canny_Edge, true, 0, count * sizeof(float), h_outputGpu.data(), NULL, &copyToHostEvent);
    Core::writeImagePGM("output_CannyEdge.pgm", h_outputGpu, imageWidth, imageHeight);
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

    // for (int i = 0; i < count; i++) h_input[i] = (rand() % 100) / 5.0f - 10.0f;
    std::vector<float> inputData;
    std::size_t inputWidth, inputHeight;
    Core::readImagePGM(filename, inputData, inputWidth, inputHeight);


    imageWidth = inputWidth - (inputWidth % wgSizeX);
    imageHeight = inputHeight - (inputHeight % wgSizeY);

    int count = imageWidth * imageHeight;
    std::vector<float> h_input(count);

    for (size_t j = 0; j < imageHeight; j++) {
        for (size_t i = 0; i < imageWidth; i++) {
            h_input[i + imageWidth * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
        }
    }


    // copyToClient
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = imageWidth;
    region[1] = imageHeight;
    region[2] = 1;

    image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);

    queue.enqueueWriteImage(
        image, true, origin, region, imageWidth * sizeof(float), 0, &(h_input[0]), NULL, &copyToClientEvent);
}
