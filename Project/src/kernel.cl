#ifndef __OPENCL_VERSION__
#    include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work (in vs go to Tools->Options->File Extensions and add 'cl' as 'C++ Microsoft Visual C++')
#else
__attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 0)))
#endif


float getValueGlobal(__read_only image2d_t image, int i, int j) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    return read_imagef(image, sampler, (int2){i, j}).x;
}



float getStrengthValue(__local float* l_strength,__global float* g_strength, int i, int j) {
    if (i < 0 || i >= WG_SIZE_X || j < 0 || j >= WG_SIZE_Y) {
        return l_strength[i * WG_SIZE_Y + j];
    }

    i = get_group_id(0) * WG_SIZE_X + i;
    j = get_group_id(1) * WG_SIZE_Y + j;
    //const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    //return read_imagef(g_strength, sampler, (int2)(i, j)).x; //fails without log

    if (i < 0 || i >= get_global_size(0) || j < 0 || j >= get_global_size(1)) return 0;
    return g_strength[i * get_global_size(1) + j];

}


/**
canny Edge Kernel with a local Buffer, that also copy the values around the workgroup Buffer
*/
__kernel void cannyEdge1(
    __read_only image2d_t h_input, __write_only image2d_t h_output, __global float* h_strength_output) {
    // copy to local memory

    size_t localSizeX = get_local_size(0);
    size_t localSizeY = get_local_size(1);


    __local float l_Image[(WG_SIZE_X + 2) * (WG_SIZE_Y + 2)]; // add also values above/lower/left/right from work group


    int l_Pos_x = get_local_id(0); // local Positins
    int l_Pos_y = get_local_id(1);
    int l_Pos = l_Pos_x * WG_SIZE_Y + l_Pos_y;

    int t_Pos_x = l_Pos_x + 1; // positions in local memory Buffer 'l_Image'
    int t_Pos_y = l_Pos_y + 1;
    int t_Size_y = WG_SIZE_Y + 2;

    int t_Pos = t_Pos_x * t_Size_y + t_Pos_y;


    l_Image[t_Pos] = getValueGlobal(h_input, get_global_id(0), get_global_id(1));

    // left
    if (l_Pos_x == 0) {
        l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y)] = getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1));
        // upper left corner
        if (l_Pos_y == 0)
            l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y - 1)] =
                getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1) - 1);
    }

    // right
    if (l_Pos_x == WG_SIZE_X) {
        l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y)] = getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1));

        // lower right
        if (l_Pos_y == WG_SIZE_Y) {
            l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y + 1)] =
                getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1) + 1);
        }
    }

    // upper
    if (l_Pos_y == 0) {
        l_Image[(t_Pos_x)*t_Size_y + (t_Pos_y - 1)] = getValueGlobal(h_input, get_global_id(0), get_global_id(1) - 1);
        // upper right
        if (l_Pos_x == WG_SIZE_X) {
            l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y - 1)] =
                getValueGlobal(h_input, get_global_id(0) + 1, get_global_id(1) - 1);
        }
    }

    // lower
    if (l_Pos_y == WG_SIZE_Y) {
        l_Image[(t_Pos_x)*t_Size_y + (t_Pos_y + 1)] = getValueGlobal(h_input, get_global_id(0), get_global_id(1) + 1);
        // lower left
        if (l_Pos_x == 0) {
            l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y + 1)] =
                getValueGlobal(h_input, get_global_id(0) - 1, get_global_id(1) + 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    // calculate the Gradient with the Sobel Operator
    float mm = l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y - 1)];
    float mp = l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y + 1)];
    float pm = l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y - 1)];
    float pp = l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y + 1)];

    float Gx = mm + 2 * l_Image[(t_Pos_x - 1) * t_Size_y + (t_Pos_y)] + mp - pm -
               2 * l_Image[(t_Pos_x + 1) * t_Size_y + (t_Pos_y)] - pp;

    float Gy = mm + 2 * l_Image[(t_Pos_x)*t_Size_y + (t_Pos_y - 1)] + pm - mp -
               2 * l_Image[(t_Pos_x)*t_Size_y + (t_Pos_y + 1)] - pp;


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
    //     h_strength_output, (int2)(get_global_id(0), get_global_id(1)), (float4)(strength, strength, strength, 1.0)); //fix read error
    h_strength_output[get_global_id(0)*get_global_size(1)+get_global_id(1)] = strength;

    int a_x, a_y;

    if (alpha >= M_PI / 8.0 && alpha < 3.0 * M_PI / 4.0) {
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


    if (strength < strengthA || strength < strengthB) { // if not the maximum Value
        write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){0, 0, 0, 1});
    } else {
        write_imagef(h_output, (int2){get_global_id(0), get_global_id(1)}, (float4){strength, strength, strength, 1});
    }
}

/**
canny Edge Kernel with a local Buffer, that overlaps with the neighbour Buffer
*/
//__kernel void cannyEdge2(__read_only image2d_t h_input, image2d_t h_output, __constant float* filter) {}
