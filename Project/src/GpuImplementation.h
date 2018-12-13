#pragma once

#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <boost/lexical_cast.hpp>

class GpuImplementation {
private:
    cl::Context createContext();
    cl::Program buildKernel(cl::Context context, int deviceNr);

    cl::CommandQueue queue;
    cl::Program program;
    cl::Context context;

    cl::Kernel nonMaxUp_Kernel;
    cl::Kernel sobel_Kernel;
    cl::Kernel gaussC_kernel;
    cl::Kernel hysterese_kernel;

    cl::Event copyToClientEvent;
    cl::Event executionEvent;
    cl::Event copyToHostEvent;

    std::size_t imageWidth, imageHeight;
    cl::Image2D image;

    std::size_t wgSizeX = 10; // ToDo no Hard coded Work Group Size
    std::size_t wgSizeY = 10;

public:
    GpuImplementation(int deviceNr = 1);
    ~GpuImplementation();

    void execute(float T1, float T2);
    void printTimeMeasurement();
    void loadImage(const boost::filesystem::path& filename);
};
