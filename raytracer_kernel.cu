#include "raytracer.h"

__global__  void raytrace(unsigned char * out, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int y = i / width;
    int x = i - y * width;

    
    out[4 * i]     = x / 2;
    out[4 * i + 1] = 0;
    out[4 * i + 2] = y / 2;
    out[4 * i + 3] = 255;
}
