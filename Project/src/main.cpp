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

#include "GpuImplementation.h"
#include "CPUImplementation.h"
#include <Core/Time.hpp>
#include <Core/Image.hpp>

/**
        Main-Method

        @param argc: start-argument
        @return Error-Code
*/
int main(int argc, char** argv) {
    GpuImplementation gpuImplementation(
        argc < 2 ? 1 : atoi(argv[1])); // if no start argument is given, use first device
    gpuImplementation.loadImage(argc < 3 ? "lena.pgm" : argv[2]);
    gpuImplementation.execute(0.0,0.7);
    gpuImplementation.printTimeMeasurement();

    CPUImplementation cpuI; // if no start argument is given, use first device
    cpuI.loadImage(argc < 3 ? "lena.pgm" : argv[2]);
    Core::TimeSpan cpuStart = Core::getCurrentTime();
    cpuI.execute(0.0,0.7);
    Core::TimeSpan cpuEnd= Core::getCurrentTime();

    Core::TimeSpan cpuExecute = cpuEnd - cpuStart;
    cpuI.printTimeMeasurement(cpuExecute);

	//check if the result image is the same
	std::size_t inputWidth, inputHeight;
	std::vector<float> inputDataCPU, inputDataGPU;
	Core::readImagePGM("output_CannyEdge_CPU.pgm", inputDataCPU, inputWidth, inputHeight);
	Core::readImagePGM("output_CannyEdge.pgm", inputDataGPU, inputWidth, inputHeight);

	std::size_t errorCount = 0;
	for (size_t i = 0; i < inputWidth; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < inputHeight; j = j + 1) { //loop in the y-direction
			size_t index = i + j * inputWidth;
			// Allow small differences between CPU and GPU results (due to different rounding behavior)
			if (!(std::abs(inputDataCPU[index] - inputDataGPU[index]) <= 1e-5)) {
				if (errorCount < 10)
					std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << inputDataGPU[index] << ", CPU value is " << inputDataCPU[index] << std::endl;
				else if (errorCount == 10)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " differences" << std::endl;
		return 1;
	}

	std::cout << std::endl;

	std::cout << "Keine relevanten Unterschiede entdeckt" << std::endl;

    return 0;
}
