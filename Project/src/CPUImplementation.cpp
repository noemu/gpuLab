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
	std::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	std::size_t<3> region;
	region[0] = imageWidth;
	region[1] = imageHeight;
	region[2] = 1;
	std::size_t size = count * sizeof (float); // Size of data in bytes

	ASSERT(imageHeight % wgSizeY == 0); // imagageWidth/height should be dividable by wgSize
	ASSERT(imageWidth % wgSizeX == 0);

	// Allocate space for output data from CPU
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);

	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);

		//////// Load input data ////////////////////////////////
		// Use random input data
		/*
		for (int i = 0; i < count; i++)
			h_input[i] = (rand() % 100) / 5.0f - 10.0f;
		*/
		// Use an image (Valve.pgm) as input data
		{
			std::vector<float> inputData;
			std::size_t inputWidth, inputHeight;
			Core::readImagePGM("Valve.pgm", inputData, inputWidth, inputHeight);
			for (size_t j = 0; j < countY; j++) {
				for (size_t i = 0; i < countX; i++) {
					h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
				}
			}
		}

	//
	// Gauss calculation (schleife 10 durchläufe)
	//

	//
	// Sobel calculation
	//

	//
	// Non Maximum Suppression
	//

	Core::writeImagePGM("output_NonMaxSup_CPU.pgm", h_outputCpu, imageWidth,
			imageHeight);

	//
	// Hysterese Kernel
	//
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
			<< "GPU-Performance" << std::setw(generalColumnWidth)
			<< "copy to client" << std::setw(generalColumnWidth)
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
	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
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

	// copyToClient
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = imageWidth;
	region[1] = imageHeight;
	region[2] = 1;

	image = cl::Image2D(context, CL_MEM_READ_WRITE,
			cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight);
}

float getValueGlobal(image2d_t image, int i, int j) {
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP
			| CLK_FILTER_NEAREST;
	return read_imagef(image, sampler, (int2 ) { i, j }).x;
}

void CPUImplementation::gaussConvolution(image2d_t h_input,
		image2d_t h_output) {
	float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add also values above/lower/left/right from work group

	int l_Pos_x = get_local_id(0); // local Positins
	int l_Pos_y = get_local_id(1);
	int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

	int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
	int t_Pos_y = l_Pos_y + 1;
	int t_Size_x = WG_SIZE_X + 2;

	// copy Values to local Buffer
	//auf rand überprüfen
	//vergleichen ob man sich am rand befindet (siehe copyToLocal)
	// bei rändern den wert rüber "kopieren" (nächst inneren wert nehmen)
	barrier (CLK_LOCAL_MEM_FENCE);

	/*
	 * calculate the Convolution with a Gauss Kernel
	 * mm = minus minus
	 * mp = minus pluss
	 * mn = minus null etc
	 */
	float mm = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)];
	float mp = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)];
	float pm = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)];
	float pp = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)];
	float mn = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)];
	float pn = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)];
	float nm = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)];
	float np = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)];
	float nn = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y)];

	float value = 1.0 / 16.0
			* (mm + mp + pm + pp + 2.0 * (nm + np + mn + mp) + 4.0 * nn);

	write_imagef(h_output, (int2 ) { get_global_id(0), get_global_id(1) },
			(float4 ) { value, value, value, 1 });
}

/**
 canny Edge Kernel with a local Buffer, that also copy the values around the workgroup Buffer
 */
void CPUImplementation::sobel1(image2d_t h_input, image2d_t h_output_Strength,
		image2d_t h_output_Direction) {
	// copy to local memory

	float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add all values and values above/lower/left/right from work group

	int l_Pos_x = get_local_id(0); // local Positins
	int l_Pos_y = get_local_id(1);
	int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

	int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
	int t_Pos_y = l_Pos_y + 1;
	int t_Size_x = WG_SIZE_X + 2;

	// copy Values to local Buffer
	copyImageToLocal(h_input, l_Image);
	barrier (CLK_LOCAL_MEM_FENCE);

	// calculate the Gradient with the Sobel Operator
	float mm = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)];
	float mp = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)];
	float pm = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)];
	float pp = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)];

	float Gx = mm + 2.0 * l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)] + mp
			- pm - 2.0 * l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)] - pp;

	float Gy = mm + 2 * l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)] + pm - mp
			- 2 * l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)] - pp;

	// edge strength
	float value = sqrt(Gx * Gx + Gy * Gy);
	write_imagef(h_output_Strength,
			(int2 ) { get_global_id(0), get_global_id(1) }, (float4 ) {
									value, value, value, 1 });

	// edget direction
	value = atan2(Gy, Gx);
	write_imagef(h_output_Direction,
			(int2 ) { get_global_id(0), get_global_id(1) }, (float4 ) {
									value, value, value, 1 });
}

void CPUImplementation::nonMaximumSuppressor(float* l_Strength,
		image2d_t h_output, float strength, int t_Pos_x, int t_Pos_y, int a_x,
		int a_y) {

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

	write_imagef(h_output, (int2 ) { get_global_id(0), get_global_id(1) },
			(float4 ) { strength, strength, strength, 1 });
}

/**
 canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
 */
void CPUImplementation::nonMaximumSuppression(image2d_t h_input_Strength,
		image2d_t h_input_Direction, image2d_t h_output) {
	float l_Strength[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add all values and values above/lower/left/right from work group

	// copy Values to local Buffer
	copyImageToLocal(h_input_Strength, l_Strength);
	barrier (CLK_LOCAL_MEM_FENCE);

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

void CPUImplementation::followEdge(int2 lastDirection, int2 pos,
		image2d_t h_input, float* h_output, float T1, float T2) {
	bool finished = false;
	int2 directions[8] = { (int2) (0, 1), (int2) (1, 0), (int2) (0, -1),
			(int2) (-1, 0), (int2) (1, 1), (int2) (-1, -1), (int2) (-1, 1),
			(int2) (1, -1) };

	// while (!finished) {

	for (int i = 0; i < get_global_size(0) + get_global_size(1); i++) { //exit criteria if an endless loop appears
		bool newValueFound = false;

		for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
			int2 direction = directions[dirIndex];
			if (direction.x == -lastDirection.x
					&& direction.y == -lastDirection.y)
				continue; // don't go backwards

			int2 newPos = pos + direction;

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

void CPUImplementation::hysterese(image2d_t h_input, float* h_output, float T1,
		float T2) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int2 pos = (int2) (x, y);

	float value = getValueGlobal(h_input, x, y);

	if (value > T2) {
		h_output[pos.x + get_global_size(0) * pos.y] = .5;

		int2 directions[8] = { (int2) (0, 1), (int2) (1, 0), (int2) (0, -1),
				(int2) (-1, 0), (int2) (1, 1), (int2) (-1, -1), (int2) (-1, 1),
				(int2) (1, -1) };

		for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
			int2 direction = directions[dirIndex];
			followEdge(direction, pos, h_input, h_output, T1, T2);
		}
	}
}
