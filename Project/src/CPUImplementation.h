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
    void printTimeMeasurement(Core::TimeSpan cpuExecutionTime);
    void loadImage(const boost::filesystem::path& filename);
	void copyImageToLocal(image2d_t h_Image, float* l_Buffer);
	void gaussConvolution(image2d_t h_input, image2d_t h_output);
	void sobel1(image2d_t h_input, image2d_t h_output_Strength, image2d_t h_output_Direction);
	void nonMaximumSuppressor(float* l_Strength, image2d_t h_output, float strength, int t_Pos_x, int t_Pos_y, int a_x, int a_y);
	void nonMaximumSuppression(image2d_t h_input_Strength, image2d_t h_input_Direction, image2d_t h_output);
	void followEdge(int2 lastDirection, int2 pos, image2d_t h_input, float* h_output, float T1, float T2);
	void hysterese(image2d_t h_input, float* h_output, float T1, float T2);
};
