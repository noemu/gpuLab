/**
        Canny-Edge Detector
        main.cpp
        Create and execute the GPU-Kernel.

        //TODO Surnames @author Ellen ..., Marvin ..., Phong ..., Rafael Jarosch
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

    return 0;
}
