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

#include "OpenGlRenderer.h"

#include "GpuImplementation.h"


/**
        Main-Method

        @param argc: start-argument
        @return Error-Code
*/
int main(int argc, char** argv) {
    GpuImplementation* gpuImplementation =
        new GpuImplementation(argc < 2 ? 1 : atoi(argv[1])); // if no start argument is given, use first device
    gpuImplementation->loadImage("lena.pgm");
    gpuImplementation->execute(0.0, 0.9);
    gpuImplementation->printTimeMeasurement();

    OpenGlRenderer::OpenGlRendererStart(argc, argv, gpuImplementation);

    return 0;
}
