#pragma once

#include <boost/lexical_cast.hpp>

class CPUImplementation {
private:

    std::size_t imageWidth, imageHeight;
    //cl::Image2D image;

    std::size_t wgSizeX = 10; // ToDo no Hard coded Work Group Size
    std::size_t wgSizeY = 10;

public:
    CPUImplementation(int deviceNr = 1);
    ~CPUImplementation();

    void execute(float T1, float T2);
    void printTimeMeasurement();
    void loadImage(const boost::filesystem::path& filename);
};
