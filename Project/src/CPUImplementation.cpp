#include "CPUImplementation.h"

#include <Core/Time.hpp>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

/**
 constructor

 @param deviceNr. The number of the device to be used
 */
CPUImplementation::CPUImplementation(int deviceNr) {
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

	int count = imageWidth * imageHeight;
	std::vector<size_t> origin(3);
	origin[0] = origin[1] = origin[2] = 0;
	std::vector<size_t> region (3);
	region[0] = imageWidth;
	region[1] = imageHeight;
	region[2] = 1;
	std::size_t size = count * sizeof (float); // Size of data in bytes

	ASSERT(imageHeight % wgSizeY == 0); // imagageWidth/height should be dividable by wgSize
	ASSERT(imageWidth % wgSizeX == 0);

	// Allocate space for output data from CPU
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_direction(count);

	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);

	//
	// Gauss calculation (schleife 10 durchläufe)
	//
	for (int i = 0; i < 10; i++) {
		gaussConvolution(h_input, h_outputCpu);
	}

	//
	// Sobel calculation
	//
	sobelHost(h_input, h_outputCpu, wgSizeX * 400, wgSizeY * 400);

	//
	// Non Maximum Suppression
	//
	nonMaximumSuppression(h_input, h_direction, h_outputCpu);

	Core::writeImagePGM("output_NonMaxSup_CPU.pgm", h_outputCpu, imageWidth,
			imageHeight);

	//
	// Hysterese Kernel
	//
	hysterese(h_input, &h_outputCpu, T1, T2);

	Core::writeImagePGM("output_CannyEdge_CPU.pgm", h_outputCpu, imageWidth,
			imageHeight);
}

//////////////////////////////////////////////////////////////////////////////
// Sobel implementation
//////////////////////////////////////////////////////////////////////////////
int CPUImplementation::getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image
float CPUImplementation::getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/**
*the sobel calculation for the host
*/
void CPUImplementation::sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, std::size_t countX, std::size_t countY) {
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
}

/**
 * print the time
 */
void CPUImplementation::printTimeMeasurement(Core::TimeSpan cpuExecutionTime) {

	std::stringstream outputString;

	const int firstColumnWidth = 24;
	const int generalColumnWidth = 19;

	outputString << std::left << std::setw(firstColumnWidth)
			//<< "GPU-Performance" << std::setw(generalColumnWidth)
			//<< "copy to client" << std::setw(generalColumnWidth)
			<< "copy to host" << std::setw(generalColumnWidth) << "execution"
			<< std::endl;
	outputString << std::left << std::setw(firstColumnWidth) << ""
			//<< std::setw(generalColumnWidth) << copyToClienTime.getSeconds()
			//<< std::setw(generalColumnWidth) << copyToHostTime.getSeconds()
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
	std::vector<float> h_input(count);

	for (size_t j = 0; j < imageHeight; j++) {
		for (size_t i = 0; i < imageWidth; i++) {
			h_input[i + imageWidth * j] = inputData[(i % inputWidth)
					+ inputWidth * (j % inputHeight)];
		}
	}
}

void CPUImplementation::gaussConvolution(std::vector<float> h_input, std::vector<float> h_output) {
	int countX = wgSizeX * 400;
	int countY = wgSizeY * 400;
	int const WG_SIZE_X = 10;
	int const WG_SIZE_Y = 10;
	int t_Size_x = wgSizeX + 2;
	float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add also values above/lower/left/right from work group

	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			/*
			* calculate the Convolution with a Gauss Kernel
			* mm = minus minus
			* mp = minus pluss
			* mn = minus null etc
			*/
			float mm = l_Image[(Gx - 1) + t_Size_x * (Gy - 1)];
			float mp = l_Image[(Gx - 1) + t_Size_x * (Gy + 1)];
			float pm = l_Image[(Gx + 1) + t_Size_x * (Gy - 1)];
			float pp = l_Image[(Gx + 1) + t_Size_x * (Gy + 1)];
			float mn = l_Image[(Gx - 1) + t_Size_x * (Gy)];
			float pn = l_Image[(Gx + 1) + t_Size_x * (Gy)];
			float nm = l_Image[(Gx)+ t_Size_x * (Gy - 1)];
			float np = l_Image[(Gx)+ t_Size_x * (Gy + 1)];
			float nn = l_Image[(Gx)+ t_Size_x * (Gy)];

			float value = 1.0 / 16.0
				* (mm + mp + pm + pp + 2.0 * (nm + np + mn + mp) + 4.0 * nn);
			h_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}

	//auf rand überprüfen
	//vergleichen ob man sich am rand befindet (siehe copyToLocal)
	// bei rändern den wert rüber "kopieren" (nächst inneren wert nehmen)

}

void CPUImplementation::nonMaximumSuppressor(float* l_Strength,
	std::vector<float> h_output, float strength, int t_Pos_x, int t_Pos_y, int a_x,
		int a_y) {


	int const WG_SIZE_X = 10;
	int const WG_SIZE_Y = 10;
	// Non Maximum Suppression
	float strengthA = l_Strength[(t_Pos_x + a_x)
			+ (WG_SIZE_X + 2) * (t_Pos_y + a_y)];
	float strengthB = l_Strength[(t_Pos_x - a_x)
			+ (WG_SIZE_X + 2) * (t_Pos_y - a_y)];

	if (strength > strengthA && strength > strengthB) { // if not the maximum Value
														// strength = 1.0;
	} else {
		strength = 0;
	}

	write_imagef(h_output, (int ) { get_global_id(0), get_global_id(1) },
			(float4 ) { strength, strength, strength, 1 });
}

/**
 canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
 */
void CPUImplementation::nonMaximumSuppression(std::vector<float> h_input_Strength,
	std::vector<float> h_input_Direction, std::vector<float> h_output) {

	int const WG_SIZE_X = 10;
	int const WG_SIZE_Y = 10;
	float l_Strength[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add all values and values above/lower/left/right from work group

	float alpha = getValueGlobal(h_input_Direction, get_global_id(0),
			get_global_id(1));

	int l_Pos_x = get_local_id(0); // local Positins
	int l_Pos_y = get_local_id(1);
	int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

	int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
	int t_Pos_y = l_Pos_y + 1;
	int t_Size_x = WG_SIZE_X + 2;
	int t_Pos = t_Pos_x + t_Size_x * t_Pos_y;

	float strength = l_Strength[t_Pos];

	float pi_8 = M_PI / 8.0;

	if ((alpha > -1.0 * pi_8 && alpha < pi_8)
			|| (alpha > 7.0 * pi_8 && alpha < M_PI)
			|| (alpha < -7.0 * pi_8 && alpha > -M_PI)) {
		// l or r
		nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y,
				1, 0);
	}
	if ((alpha > pi_8 && alpha < 3.0 * pi_8)
			|| (alpha >= -7.0 * pi_8 && alpha < -5.0 * pi_8)) {
		// tr(top right) or bl(bottom left
		nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y,
				1, 1);
	}
	if ((alpha > 3.0 * pi_8 && alpha < 5.0 * pi_8)
			|| (alpha > -5.0 * pi_8 && alpha < -3.0 * pi_8)) {
		// t or b
		nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y,
				0, 1);
	}
	if ((alpha > 5.0 * pi_8 && alpha < 7.0 * pi_8)
			|| (alpha > -3.0 * pi_8 && alpha < -1.0 * pi_8)) {
		// tl or br
		nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y,
				-1, 1);
	}
}

void CPUImplementation::followEdge(int lastDirection, int pos,
	std::vector<float> h_input, float* h_output, float T1, float T2) {
	bool finished = false;
	int directions[8] = { (int) (0, 1), (int) (1, 0), (int) (0, -1),
			(int) (-1, 0), (int) (1, 1), (int) (-1, -1), (int) (-1, 1),
			(int) (1, -1) };

	// while (!finished) {

	for (int i = 0; i < get_global_size(0) + get_global_size(1); i++) { //exit criteria if an endless loop appears
		bool newValueFound = false;

		for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
			int direction = directions[dirIndex];
			if (direction.x == -lastDirection.x
					&& direction.y == -lastDirection.y)
				continue; // don't go backwards

			int newPos = pos + direction;

			if (newPos.x < 0 || newPos.x >= get_global_size(0) || newPos.y < 0
					|| newPos.y >= get_global_size(1)) // skip out of bound Values
				continue;

			if (h_output[newPos.x + get_global_size(0) * newPos.y] != 0)
				continue; //skip already written edges. slow global memory access

			float nextValue = getValueGlobal(h_input, newPos.x, newPos.y);

			if (nextValue > T2)
				continue; // skip values that are computed by other Threads

			if (nextValue > T1) {
				lastDirection = direction;
				pos = newPos;
				newValueFound = true;
				h_output[newPos.x + get_global_size(0) * newPos.y] = .5;
				break;
			}
		}

		if (!newValueFound) {
			finished = true;
			break;
		};
	}
	if (!finished)
		h_output[pos.x + get_global_size(0) * pos.y] = 1;
}

void CPUImplementation::hysterese(std::vector<float> h_input, float* h_output, float T1, float T2) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int pos = (int) (x, y);

	float value = getValueGlobal(h_input, x, y);

	if (value > T2) {
		h_output[pos.x + get_global_size(0) * pos.y] = .5;

		int directions[8] = { (int) (0, 1), (int) (1, 0), (int) (0, -1),
				(int) (-1, 0), (int) (1, 1), (int) (-1, -1), (int) (-1, 1),
				(int) (1, -1) };

		for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
			int direction = directions[dirIndex];
			followEdge(direction, pos, h_input, h_output, T1, T2);
		}
	}
}
