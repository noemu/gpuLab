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

/**
        Main-Method

        @param argc: start-argument
        @return Error-Code
*/
int main(int argc, char** argv) {
    GpuImplementation gpuImplementation(
        argc < 2 ? 1 : atoi(argv[1])); // if no start argument is given, use first device
    gpuImplementation.loadImage(argc < 3 ? "lena.pgm" : argv[2]);
    gpuImplementation.execute(0.0,0.9);
    gpuImplementation.printTimeMeasurement();

    CPUImplementation cpuImplementation(
        argc < 2 ? 1 : atoi(argv[1])); // if no start argument is given, use first device
    cpuImplementation.loadImage(argc < 3 ? "lena.pgm" : argv[2]);
    Core::TimeSpan cpuStart = Core::getCurrentTime();
    cpuImplementation.execute(0.0,0.9);
    Core::TimeSpan cpuEnd= Core::getCurrentTime();

    Core::TimeSpan cpuExecute = cpuEnd - cpuStart;
    cpuImplementation.printTimeMeasurement(cpuExecute);

    return 0;
}
