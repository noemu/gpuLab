#include "CPUImplementation.h"

#include <Core/Assert.hpp>
#include <Core/Image.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

/**
 constructor

 @param deviceNr. The number of the device to be used
 */
CPUImplementation::CPUImplementation() {
	imageHeight = 0;
	imageWidth = 0;
}

/**
 destructor
 */
CPUImplementation::~CPUImplementation() {
}

/**
 executes the Client
 */
void CPUImplementation::execute(float T1 = 0.1, float T2 = 0.7) {

	count = imageWidth * imageHeight;
	std::vector<size_t> origin(3);
	origin[0] = origin[1] = origin[2] = 0;
	std::vector<size_t> region(3);
	region[0] = imageWidth;
	region[1] = imageHeight;
	region[2] = 1;
	std::size_t size = count * sizeof(float); // Size of data in bytes

	ASSERT(imageHeight % wgSizeY == 0); // imagageWidth/height should be dividable by wgSize
	ASSERT(imageWidth % wgSizeX == 0);

	// Allocate space for output data from CPU
	std::vector<float> h_outputCpu(count);
	std::vector<float> h_direction(count);
	std::vector<float> h_magnitude(count);
	std::vector<float> non_max_sup(count);

	memset(h_outputCpu.data(), 0, size);
	memset(h_direction.data(), 255, size);
	memset(h_magnitude.data(), 255, size);
	memset(non_max_sup.data(), 255, size);

	//
	// Gauss calculation (schleife 10 durchläufe)
	//

	for (int i = 0; i < 10; i++) {
		std::vector<float> gaussOut = CPUImplementation::gaussConvolution();
		h_input = gaussOut;
	}
	Core::writeImagePGM("output_Gauss_CPU.pgm", h_input, imageWidth,
			imageHeight);

	//
	// Sobel calculation
	// wgSizen raus hauen
	sobelHost(h_direction.data(), h_magnitude.data());

	Core::writeImagePGM("output_Sobel_Mag_CPU.pgm", h_magnitude, imageWidth,
			imageHeight);
	//
	// Non Maximum Suppression
	//
	nonMaximumSuppression(h_magnitude, h_direction, non_max_sup.data());

	Core::writeImagePGM("output_NonMaxSup_CPU.pgm", non_max_sup, imageWidth,
			imageHeight);
	//
	// Hysterese Kernel
	//
	h_input.assign(&non_max_sup[0], &non_max_sup[count - 1]);
	hysterese(h_outputCpu.data(), T1, T2);

	Core::writeImagePGM("output_CannyEdge_CPU.pgm", h_outputCpu, imageWidth,
			imageHeight);
}

/**
 * print the time
 */
void CPUImplementation::printTimeMeasurement(Core::TimeSpan cpuExecutionTime) {

	std::stringstream outputString;

	const int firstColumnWidth = 24;
	const int generalColumnWidth = 19;

	outputString << std::left << std::setw(firstColumnWidth)
			<< "CPU-Performance" << std::setw(generalColumnWidth) << "execution"
			<< std::endl;
	outputString << std::left << std::setw(firstColumnWidth) << ""
			<< std::setw(generalColumnWidth) << cpuExecutionTime.getSeconds()
			<< std::endl;

	std::cout << outputString.str() << std::endl;
}

void CPUImplementation::loadImage(const boost::filesystem::path& filename) {

	// for (int i = 0; i < count; i++) h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	std::size_t inputWidth, inputHeight;
	std::vector<float> inputData;
	Core::readImagePGM(filename, inputData, inputWidth, inputHeight);

	imageWidth = inputWidth - (inputWidth % wgSizeX);
	imageHeight = inputHeight - (inputHeight % wgSizeY);

	int count = imageWidth * imageHeight;
	h_input.resize(count);
	h_input.reserve(count);

	for (int j = 0; j < imageHeight; j++) {
		for (int i = 0; i < imageWidth; i++) {
			h_input[i + imageWidth * j] = inputData[(i % inputWidth)
					+ inputWidth * (j % inputHeight)];
		}
	}
}

// Read value from global array a, return 0 if outside image
float CPUImplementation::getValueGlobal(const std::vector<float>& a, int x,
		int y) {
	if (x < 0)
		x = 0;
	if (x >= imageWidth)
		x = imageWidth - 1;
	if (y < 0)
		y = 0;
	if (y >= imageHeight)
		y = imageHeight - 1;

	return a[x + imageWidth * y];
}

/**
 *the sobel calculation for the host
 */
void CPUImplementation::sobelHost(float* h_direction, float* h_outputCpu) {
	for (int i = 0; i < imageWidth; i++) {
		for (int j = 0; j < imageHeight; j++) {
			float Gx = getValueGlobal(h_input, i - 1, j - 1)
					+ 2 * getValueGlobal(h_input, i - 1, j)
					+ getValueGlobal(h_input, i - 1, j + 1)
					- getValueGlobal(h_input, i + 1, j - 1)
					- 2 * getValueGlobal(h_input, i + 1, j)
					- getValueGlobal(h_input, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, i - 1, j - 1)
					+ 2 * getValueGlobal(h_input, i, j - 1)
					+ getValueGlobal(h_input, i + 1, j - 1)
					- getValueGlobal(h_input, i - 1, j + 1)
					- 2 * getValueGlobal(h_input, i, j + 1)
					- getValueGlobal(h_input, i + 1, j + 1);
			h_outputCpu[i + imageWidth * j] = sqrt(Gx * Gx + Gy * Gy);
			// edget direction
			h_direction[i + imageWidth * j] = atan2(Gy, Gx);
		}
	}
}

std::vector<float> CPUImplementation::gaussConvolution() {

	std::vector<float> h_output(count);
	h_output.reserve(count);

	//Gx durch x ersetzen Gy analog nur hier!
	for (int Gx = 0; Gx < imageWidth; Gx++) {
		for (int Gy = 0; Gy < imageHeight; Gy++) {
			/*
			 * calculate the Convolution with a Gauss Kernel
			 * mm = minus minus
			 * mp = minus pluss
			 * mn = minus null etc
			 */
			float mm = getValueGlobal(h_input, Gx - 1, Gy - 1);
			float mp = getValueGlobal(h_input, Gx - 1, Gy + 1);
			float pm = getValueGlobal(h_input, Gx + 1, Gy - 1);
			float pp = getValueGlobal(h_input, Gx + 1, Gy + 1);
			float mn = getValueGlobal(h_input, Gx - 1, Gy);
			float pn = getValueGlobal(h_input, Gx + 1, Gy);
			float nm = getValueGlobal(h_input, Gx, Gy - 1);
			float np = getValueGlobal(h_input, Gx, Gy + 1);
			float nn = getValueGlobal(h_input, Gx, Gy);

			float value =
					1.0 / 16.0
							* (mm + mp + pm + pp + 2.0 * (nm + np + mn + mp)
									+ 4.0 * nn);
			h_output[Gx + imageWidth * Gy] = value;
		}
	}
	return h_output;
}

void CPUImplementation::nonMaximumSuppressor(std::vector<float> l_Strength,
		float* h_output, float strength, int x, int y, int a_x, int a_y) {

	// Non Maximum Suppression
	float strengthA = l_Strength[(x + a_x) + (imageWidth) * (y + a_y)];
	float strengthB = l_Strength[(x - a_x) + (imageWidth) * (y - a_y)];

	if (strength > strengthA && strength > strengthB) { // if not the maximum Value
														// strength = 1.0;
	} else {
		strength = 0;
	}

	h_output[x + imageWidth * y] = strength;
}

/**
 canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
 */
void CPUImplementation::nonMaximumSuppression(
		std::vector<float> h_input_Strength,
		std::vector<float> h_input_Direction, float* h_output) {
	for (int x = 0; x < imageWidth; x++) {
		for (int y = 0; y < imageHeight; y++) {
			float alpha = getValueGlobal(h_input_Direction, x, y);

			float strength = getValueGlobal(h_input_Strength, x, y);

			float pi_8 = M_PI / 8.0;

			if ((alpha > -1.0 * pi_8 && alpha < pi_8)
					|| (alpha > 7.0 * pi_8 && alpha < M_PI)
					|| (alpha < -7.0 * pi_8 && alpha > -M_PI)) {
				// l or r
				nonMaximumSuppressor(h_input_Strength, h_output, strength, x, y,
						1, 0);
			}
			if ((alpha > pi_8 && alpha < 3.0 * pi_8)
					|| (alpha >= -7.0 * pi_8 && alpha < -5.0 * pi_8)) {
				// tr(top right) or bl(bottom left
				nonMaximumSuppressor(h_input_Strength, h_output, strength, x, y,
						1, 1);
			}
			if ((alpha > 3.0 * pi_8 && alpha < 5.0 * pi_8)
					|| (alpha > -5.0 * pi_8 && alpha < -3.0 * pi_8)) {
				// t or b
				nonMaximumSuppressor(h_input_Strength, h_output, strength, x, y,
						0, 1);
			}
			if ((alpha > 5.0 * pi_8 && alpha < 7.0 * pi_8)
					|| (alpha > -3.0 * pi_8 && alpha < -1.0 * pi_8)) {
				// tl or br
				nonMaximumSuppressor(h_input_Strength, h_output, strength, x, y,
						-1, 1);
			}
		}
	}
}

void CPUImplementation::followEdge(int2 lastDirection, int2 pos,
		float* h_output, float T1, float T2) {
	bool finished = false;
	int2 directions[8] = { int2(0, 1), int2(1, 0), int2(0, -1), int2(-1, 0),
			int2(1, 1), int2(-1, -1), int2(-1, 1), int2(1, -1) };

// while (!finished) {

	for (int i = 0; i < imageWidth + imageHeight; i++) { //exit criteria if an endless loop appears
		bool newValueFound = false;

		for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
			int2 direction = directions[dirIndex];
			if (direction.x == -lastDirection.x
					&& direction.y == -lastDirection.y)
				continue; // don't go backwards

			int2 newPos = pos + direction;

			if (newPos.x < 0 || newPos.x >= imageWidth || newPos.y < 0
					|| newPos.y >= imageHeight) // skip out of bound Values
				continue;

			if (h_output[newPos.x + imageWidth * newPos.y] != 0)
				continue; //skip already written edges. slow global memory access

			float nextValue = getValueGlobal(h_input, newPos.x, newPos.y);

			if (nextValue > T2)
				continue; // skip values that are computed by other Threads

			if (nextValue > T1) {
				lastDirection = direction;
				pos = newPos;
				newValueFound = true;
				h_output[newPos.x + imageWidth * newPos.y] = .5;
				break;
			}
		}

		if (!newValueFound) {
			finished = true;
			break;
		};
	}
	if (!finished)
		h_output[pos.x + imageWidth * pos.y] = 1;
}

void CPUImplementation::hysterese(float* h_output, float T1, float T2) {
	for(int x = 0; x < imageWidth; x++){
		for(int y = 0; y < imageHeight; y++){
			int2 pos = int2(x, y);

				float value = getValueGlobal(h_input, x, y);

				if (value > T2) {
					h_output[pos.x + imageWidth * pos.y] = .5;

					int2 directions[8] = { int2(0, 1), int2(1, 0), int2(0, -1), int2(-1, 0),
							int2(1, 1), int2(-1, -1), int2(-1, 1), int2(1, -1) };

					for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
						int2 direction = directions[dirIndex];
						followEdge(direction, pos, h_output, T1, T2);
					}
				}
		}
	}
}
