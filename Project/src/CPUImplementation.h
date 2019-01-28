#pragma once

#include "int2.h"

#include <Core/Time.hpp>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

#include <boost/filesystem.hpp>
#include <vector>

class CPUImplementation {
private:

	int count;

    int imageWidth, imageHeight;

    std::vector<float> h_input;

    int wgSizeX = 10; // ToDo no Hard coded Work Group Size
    int wgSizeY = 10;

public:
    CPUImplementation();
    ~CPUImplementation();

    void execute(float T1, float T2);


    void loadImage(const boost::filesystem::path& filename);

    void printTimeMeasurement(Core::TimeSpan cpuExecutionTime);

    std::vector<float> gaussConvolution();
	void nonMaximumSuppressor(std::vector<float> l_Strength, float* h_output, float strength, int t_Pos_x, int t_Pos_y, int a_x, int a_y);
	void nonMaximumSuppression(std::vector<float> h_input_Strength, std::vector<float> h_input_Direction, float* h_output);
	void followEdge(int2 lastDirection, int2 pos, float* h_output, float T1, float T2);
	void hysterese(float* h_output, float T1, float T2);
	float getValueGlobal(const std::vector<float>& a, int x, int y);
	void sobelHost(float* h_direction, float* h_outputCpu);
};
