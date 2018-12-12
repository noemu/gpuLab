#ifndef __OPENCL_VERSION__
#    include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work (in vs go to Tools->Options->File Extensions and add 'cl' as 'C++ Microsoft Visual C++')
#else
__attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 0)))
#endif


float getValueGlobal(__read_only image2d_t image, int i, int j) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    return read_imagef(image, sampler, (int2){i, j}).x;
}


/**
a local Buffer, that also copy the values around the workgroup Buffer
*/
void copyImageToLocal(__read_only image2d_t h_Image, __local float* l_Buffer) {

    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;

    int t_Pos = t_Pos_x + t_Size_x * t_Pos_y;

    // fill local Buffer
    l_Buffer[t_Pos] = getValueGlobal(h_Image, get_global_id(0), get_global_id(1));

    // add outer border to local buffer
    float test = getValueGlobal(h_Image, get_global_id(0), get_global_id(1));
    // left side
    if (l_Pos_x == 0) {
        // left edge
        l_Buffer[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)] =
            getValueGlobal(h_Image, get_global_id(0) - 1, get_global_id(1));
        // lower left corner
        if (l_Pos_y == 0) {
            l_Buffer[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)] =
                getValueGlobal(h_Image, get_global_id(0) - 1, get_global_id(1) - 1);
        }
        // upper left corner
        if (l_Pos_y == WG_SIZE_Y - 1) {
            l_Buffer[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)] =
                getValueGlobal(h_Image, get_global_id(0) - 1, get_global_id(1) + 1);
        }
    }
    // right side
    if (l_Pos_x == WG_SIZE_X - 1) {
        // right edge
        l_Buffer[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)] =
            getValueGlobal(h_Image, get_global_id(0) + 1, get_global_id(1));
        // lower right corner
        if (l_Pos_y == 0) {
            l_Buffer[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)] =
                getValueGlobal(h_Image, get_global_id(0) + 1, get_global_id(1) - 1);
        }
        // upper right corner
        if (l_Pos_y == WG_SIZE_Y - 1) {
            l_Buffer[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)] =
                getValueGlobal(h_Image, get_global_id(0) + 1, get_global_id(1) + 1);
        }
    }
    // lower edge
    if (l_Pos_y == 0) {
        l_Buffer[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)] =
            getValueGlobal(h_Image, get_global_id(0), get_global_id(1) - 1);
    }
    // upper edge
    if (l_Pos_y == WG_SIZE_Y - 1) {
        l_Buffer[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)] =
            getValueGlobal(h_Image, get_global_id(0), get_global_id(1) + 1);
    }
}


__kernel void gaussConvolution(__read_only image2d_t h_input, __write_only image2d_t h_output) {
    __local float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add also values above/lower/left/right from work group


    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;

    // copy Values to local Buffer
    copyImageToLocal(h_input, l_Image);
    barrier(CLK_LOCAL_MEM_FENCE);


    // calculate the Convolution with a Gauss Kernel
    float mm = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)];
    float mp = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)];
    float pm = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)];
    float pp = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)];
    float mn = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)];
    float pn = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)];
    float nm = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)];
    float np = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)];
    float nn = l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y)];

    float value = 1.0 / 16.0 * (mm + mp + pm + pp + 2.0 * (nm + np + mn + mp) + 4.0 * nn);

    write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){value, value, value, 1});
}


/**
canny Edge Kernel with a local Buffer, that also copy the values around the workgroup Buffer
*/
__kernel void sobel1(__read_only image2d_t h_input, __write_only image2d_t h_output_Strength,
    __write_only image2d_t h_output_Direction) {
    // copy to local memory

    __local float
        l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add all values and values above/lower/left/right from work group


    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;

    // copy Values to local Buffer
    copyImageToLocal(h_input, l_Image);
    barrier(CLK_LOCAL_MEM_FENCE);


    // calculate the Gradient with the Sobel Operator
    float mm = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)];
    float mp = l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)];
    float pm = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)];
    float pp = l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)];

    float Gx = mm + 2.0 * l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)] + mp - pm -
               2.0 * l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)] - pp;

    float Gy = mm + 2 * l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)] + pm - mp -
               2 * l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)] - pp;


    // edge strength
    float value = sqrt(Gx * Gx + Gy * Gy);
    write_imagef(h_output_Strength, (int2){get_global_id(0), get_global_id(1)}, (float4){value, value, value, 1});

    // edget direction
    value = atan2(Gy, Gx);
    write_imagef(h_output_Direction, (int2){get_global_id(0), get_global_id(1)}, (float4){value, value, value, 1});
}


void nonMaximumSuppressor(__local float* l_Strength, __write_only image2d_t h_output, float strength, int t_Pos_x,
    int t_Pos_y, int a_x, int a_y) {

    // Non Maximum Suppression
    float strengthA = l_Strength[(t_Pos_x + a_x) + (WG_SIZE_X + 2) * (t_Pos_y + a_y)];
    float strengthB = l_Strength[(t_Pos_x - a_x) + (WG_SIZE_X + 2) * (t_Pos_y - a_y)];


    if (strength > strengthA && strength > strengthB) { // if not the maximum Value
        // strength = 1.0;
    } else {
        strength = 0;
    }


    write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){strength, strength, strength, 1});
}

/**
canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
*/
__kernel void nonMaximumSuppression(
    __read_only image2d_t h_input_Strength, __read_only image2d_t h_input_Direction, __write_only image2d_t h_output) {

    __local float l_Strength[(WG_SIZE_X + 2) *
                             (WG_SIZE_Y + 2)]; // add all values and values above/lower/left/right from work group

    // copy Values to local Buffer
    copyImageToLocal(h_input_Strength, l_Strength);
    barrier(CLK_LOCAL_MEM_FENCE);

    float alpha = getValueGlobal(h_input_Direction, get_global_id(0), get_global_id(1));

    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;
    int t_Pos = t_Pos_x + t_Size_x * t_Pos_y;

    float strength = l_Strength[t_Pos];

    float pi_8 = M_PI / 8.0;


    if ((alpha > -1.0 * pi_8 && alpha < pi_8) || (alpha > 7.0 * pi_8 && alpha < M_PI) ||
        (alpha < -7.0 * pi_8 && alpha > -M_PI)) {
        // l or r
        nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y, 1, 0);
    }
    if ((alpha > pi_8 && alpha < 3.0 * pi_8) || (alpha >= -7.0 * pi_8 && alpha < -5.0 * pi_8)) {
        // tr(top right) or bl(bottom left
        nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y, 1, 1);
    }
    if ((alpha > 3.0 * pi_8 && alpha < 5.0 * pi_8) || (alpha > -5.0 * pi_8 && alpha < -3.0 * pi_8)) {
        // t or b
        nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y, 0, 1);
    }
    if ((alpha > 5.0 * pi_8 && alpha < 7.0 * pi_8) || (alpha > -3.0 * pi_8 && alpha < -1.0 * pi_8)) {
        // tl or br
        nonMaximumSuppressor(l_Strength, h_output, strength, t_Pos_x, t_Pos_y, -1, 1);
    }
}

void followEdge(
    int2 lastDirection, int2 pos, __read_only image2d_t h_input, __write_only image2d_t h_output, float T1, float T2) {
    bool finished = false;
    int2 directions[8] = {(int2)(0, 1), (int2)(1, 0), (int2)(0, -1), (int2)(-1, 0), (int2)(1, 1), (int2)(-1, -1),
        (int2)(-1, 1), (int2)(1, -1)};

    // while (!finished) {

    for (int i = 0; i < get_global_size(0) + get_global_size(1); i++) { //exit criteria if an endless loop appears
        bool newValueFound = false;

        for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
            int2 direction = directions[dirIndex];
            if (direction.x == -lastDirection.x && direction.y == -lastDirection.y) continue; // don't go backwards

            int2 newPos = pos + direction;

            if (newPos.x < 0 || newPos.x >= get_global_size(0) || newPos.y < 0 ||
                newPos.y >= get_global_size(1)) // skip out of bound Values
                continue;

            float nextValue = getValueGlobal(h_input, newPos.x, newPos.y);


            if (nextValue > T2) continue; // skip values that are computed by other Threads

            if (nextValue > T1) {
                lastDirection = direction;
                pos = newPos;
                newValueFound = true;
                write_imagef(h_output, newPos, (float4)(0.5, 0.5, 0.5, 1));
                break;
            }
        }

        if (!newValueFound) {
            finished = true;
            break;
        };
    }
    if (!finished) write_imagef(h_output, pos, (float4)(1, 1, 1, 1));
}


__kernel void hysterese(__read_only image2d_t h_input, __write_only image2d_t h_output, float T1, float T2) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int2 pos = (int2)(x, y);

    float value = getValueGlobal(h_input, x, y);


    if (value > T2) {
        write_imagef(h_output, pos, (float4)(.5, .5, .5, 1));

        int2 directions[8] = {(int2)(0, 1), (int2)(1, 0), (int2)(0, -1), (int2)(-1, 0), (int2)(1, 1), (int2)(-1, -1),
            (int2)(-1, 1), (int2)(1, -1)};

        for (int dirIndex = 0; dirIndex < 8; dirIndex++) {
            int2 direction = directions[dirIndex];
            followEdge(direction, pos, h_input, h_output, T1, T2);
        }
    }
}
