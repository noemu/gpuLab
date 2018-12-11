#ifndef __OPENCL_VERSION__
#    include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work (in vs go to Tools->Options->File Extensions and add 'cl' as 'C++ Microsoft Visual C++')
#else
__attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 0)))
#endif


float getValueGlobal(__read_only image2d_t image, int i, int j) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    return read_imagef(image, sampler, (int2){i, j}).x;
}


float getStrengthValue(__local float* l_strength, __global float* g_strength, int i, int j) {
    if (i < 0 || i >= WG_SIZE_X || j < 0 || j >= WG_SIZE_Y) {
        return l_strength[i + WG_SIZE_X * j];
    }

    i = get_group_id(0) * WG_SIZE_X + i;
    j = get_group_id(1) * WG_SIZE_Y + j;
    // const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    // return read_imagef(g_strength, sampler, (int2)(i, j)).x; //fails without log

    if (i < 0 || i >= get_global_size(0) || j < 0 || j >= get_global_size(1)) return 0;
    return g_strength[j * get_global_size(0) + i];
}


/**
a local Buffer, that also copy the values around the workgroup Buffer
*/
void copyImageToLocal(__local float* l_Image, __read_only image2d_t h_input) {

    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;

    int t_Pos = t_Pos_x + t_Size_x * t_Pos_y;

    // fill local Buffer
    l_Image[t_Pos] = getValueGlobal(h_input, get_global_id(0), get_global_id(1));

    // add outer border to local buffer
    float test = getValueGlobal(h_input, get_global_id(0), get_global_id(1));
    // left side
    if (l_Pos_x == 0) {
        // left edge
        l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y)] = getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1));
        // lower left corner
        if (l_Pos_y == 0) {
            l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y - 1)] =
                getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1) - 1);
        }
        // upper left corner
        if (l_Pos_y == WG_SIZE_Y - 1) {
            l_Image[(t_Pos_x - 1) + t_Size_x * (t_Pos_y + 1)] =
                getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1) + 1);
        }
    }
    // right side
    if (l_Pos_x == WG_SIZE_X - 1) {
        // right edge
        l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y)] = getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1));
        // lower right corner
        if (l_Pos_y == 0) {
            l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y - 1)] =
                getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1) - 1);
        }
        // upper right corner
        if (l_Pos_y == WG_SIZE_Y - 1) {
            l_Image[(t_Pos_x + 1) + t_Size_x * (t_Pos_y + 1)] =
                getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1) + 1);
        }
    }
    // lower edge
    if (l_Pos_y == 0) {
        l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y - 1)] = getValueGlobal(h_input, get_global_id(0), get_global_id(1) - 1);
    }
    // upper edge
    if (l_Pos_y == WG_SIZE_Y - 1) {
        l_Image[(t_Pos_x) + t_Size_x * (t_Pos_y + 1)] = getValueGlobal(h_input, get_global_id(0), get_global_id(1) + 1);
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
    copyImageToLocal(l_Image, h_input);
    barrier(CLK_LOCAL_MEM_FENCE);


     //calculate the Convolution with a Gauss Kernel
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
__kernel void cannyEdge1(
    __read_only image2d_t h_input, __write_only image2d_t h_output, __global float* h_strength_output) {
    // copy to local memory

    __local float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add also values above/lower/left/right from work group


    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x + WG_SIZE_X + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_x = WG_SIZE_X + 2;

	//copy Values to local Buffer
    copyImageToLocal(l_Image, h_input);
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

    float strength = sqrt(Gx * Gx + Gy * Gy);

    // edget direction
    float alpha = atan2(Gy, Gx);
    alpha = fabs(alpha);


    // save strength localy
    __local float l_Strength[(WG_SIZE_X) * (WG_SIZE_Y)];
    l_Strength[l_Pos] = strength;
    barrier(CLK_LOCAL_MEM_FENCE);

    // save strength globally
    // write_imagef(
    //     h_strength_output, (int2)(get_global_id(0), get_global_id(1)), (float4)(strength, strength, strength, 1.0));
    //     //fix read error
    h_strength_output[get_global_id(1) * get_global_size(0) + get_global_id(0)] = strength;

    int a_x, a_y;

    if (alpha >= M_PI / 8.0 && alpha < 3.0 * M_PI / 8.0) {
        // tr(top right) or bl(bottom left
        a_x = 1;
        a_y = 1;

    } else {
        if (alpha >= 3.0 * M_PI / 8.0 && alpha < 5.0 * M_PI / 8.0) {
            // t or b
            a_x = 0;
            a_y = 1;
        } else {
            if (alpha >= 5.0 * M_PI / 8.0 && alpha < 7.0 * M_PI / 8.0) {
                // tl or br
                a_x = -1;
                a_y = 1;
            } else {
                // l or r
                a_x = 1;
                a_y = 0;
            }
        }
    }


    // Non Maximum Suppression
    float strengthA = getStrengthValue(l_Strength, h_strength_output, l_Pos_x + a_x, l_Pos_y + a_y);
    float strengthB = getStrengthValue(l_Strength, h_strength_output, l_Pos_x - a_x, l_Pos_y - a_y);


    if (strength < strengthA || strength <= strengthB) { // if not the maximum Value
        write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){0, 0, 0, 1});
    } else {
        write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){strength, strength, strength, 1});
    }
}

/**
canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
*/
//__kernel void cannyEdge2(__read_only image2d_t h_input, image2d_t h_output, __constant float* filter) {}
