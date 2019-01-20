#pragma once

#include <Core/Time.hpp>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

class CPUImplementation {
private:

    std::size_t imageWidth, imageHeight;

    std::size_t wgSizeX = 10; // ToDo no Hard coded Work Group Size
    std::size_t wgSizeY = 10;

public:
    CPUImplementation(int deviceNr = 1);
    ~CPUImplementation();

    void execute(float T1, float T2);
    void printTimeMeasurement(Core::TimeSpan cpuExecutionTime);
    void loadImage(const boost::filesystem::path& filename);
	void copyImageToLocal(std::vector<float> h_Image, float* l_Buffer);
	void gaussConvolution(std::vector<float> h_input, std::vector<float> h_output);
	void sobel1(std::vector<float> h_input, std::vector<float> h_output_Strength, std::vector<float> h_output_Direction);
	void nonMaximumSuppressor(float* l_Strength, std::vector<float> h_output, float strength, int t_Pos_x, int t_Pos_y, int a_x, int a_y);
	void nonMaximumSuppression(std::vector<float> h_input_Strength, std::vector<float> h_input_Direction, std::vector<float> h_output);
	void followEdge(int2 lastDirection, int2 pos, std::vector<float> h_input, float* h_output, float T1, float T2);
	void hysterese(std::vector<float> h_input, float* h_output, float T1, float T2);
};
