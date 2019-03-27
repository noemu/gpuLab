/**
        Canny-Edge Detector
        main.cpp
        Create and execute the GPU-Kernel and the Host.

        //TODO Surnames @author Ellen ..., Marvin Knodel 3229587, Phong ..., Rafael Jarosch
*/

/**#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>*/

#include "OpenGlRenderer.h"

#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include "CPUImplementation.h"
#include "GpuImplementation.h"

void compare(std::string cpuImg, std::string gpuImg, double e) {

    // check if the result image is the same
    std::size_t inputWidth, inputHeight;
    std::vector<float> inputDataCPU, inputDataGPU;
    Core::readImagePGM(cpuImg, inputDataCPU, inputWidth, inputHeight);
    Core::readImagePGM(gpuImg, inputDataGPU, inputWidth, inputHeight);
    std::size_t errorCount = 0;
    for (size_t i = 0; i < inputWidth; i = i + 1) {
        // loop in the x-direction
        for (size_t j = 0; j < inputHeight; j = j + 1) {
            // loop in the y-direction
            size_t index = i + j * inputWidth;
            // Allow small differences between CPU and GPU results (due to different rounding behavior)
            if (!(std::abs(inputDataCPU[index] - inputDataGPU[index]) <= e)) {
                if (errorCount < 10)
                    std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << inputDataGPU[index]
                              << ", CPU value is " << inputDataCPU[index] << std::endl;
                else if (errorCount == 10)
                    std::cout << "..." << std::endl;

                errorCount++;
            }
        }
    }
    if (errorCount != 0) {
        std::cout << "Found " << errorCount << " differences" << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "No relevant differences discovered" << std::endl;
    }
}


/**
        Main-Method

        @param argc: start-argument
        @return Error-Code
*/
int main(int argc, char** argv) {

    float T1 = 0.0;
    float T2 = 0.7;
    std::string imageName = "lena.pgm";


    GpuImplementation* gpuImplementation =
        new GpuImplementation(argc < 2 ? 1 : atoi(argv[1])); // if no start argument is given, use first device
    gpuImplementation->loadImage(argc < 3 ? imageName : argv[2]);
    gpuImplementation->execute(T1, T2);
    gpuImplementation->printTimeMeasurement();


    bool runCpu = true;

    if (runCpu) {
        CPUImplementation cpuI; // if no start argument is given, use first device
        cpuI.loadImage(argc < 3 ? imageName : argv[2]);
        Core::TimeSpan cpuStart = Core::getCurrentTime();
        cpuI.execute(T1, T2);
        Core::TimeSpan cpuEnd = Core::getCurrentTime();

        Core::TimeSpan cpuExecute = cpuEnd - cpuStart;
        cpuI.printTimeMeasurement(cpuExecute);

        std::cout << std::endl << "Comparison: Gauss" << std::endl;
        compare("output_Gauss_CPU.pgm", "output_GaussSmoothed.pgm", 1e-1);

        std::cout << std::endl << "Comparison: Gradient" << std::endl;
        compare("output_Gradient_CPU.pgm", "output_Gradient.pgm", 1e-1);

        std::cout << std::endl << "Comparison: Non Max Suppression" << std::endl;
        compare("output_NonMaxSup_CPU.pgm", "output_NonMaxSup.pgm", 1e-1);

        std::cout << std::endl << "Comparison: Canny Edge" << std::endl;
        compare("output_CannyEdge_CPU.pgm", "output_CannyEdge.pgm", 1e-1);
    }


        std::cout << std::endl << "press up arrow to add more lines (down for less). Press right arrow for longer lines (left for shorter) " << std::endl;

    OpenGlRenderer::OpenGlRendererStart(argc, argv, gpuImplementation);

    return 0;
}
