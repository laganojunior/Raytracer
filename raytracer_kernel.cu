#include "raytracer.h"
#include "cutil_math.h"

struct Ray
{
    float3 o;
    float3 d; 
};


/////////////////////////////////////////////////////////////////////////////
// Function to calculate the intersection point of a ray with a sphere. 
// Assumes the ray has unit length
////////////////////////////////////////////////////////////////////////////
__device__ bool rayIntersectSphere(Ray ray, float3 center, float radius, float3 * result)
{
    float3 oMinusC = ray.o - center; 
    float  dDotOC  = dot(ray.d, oMinusC);
    float disc = dDotOC * dDotOC - (dot(oMinusC, oMinusC) - radius * radius);

    if (disc < 0)
        return false;

    disc = sqrt(disc);

    if (-dDotOC - disc > 0)
    {
        *result = ray.o - (-dDotOC + disc) * ray.d;
        return true;
    }

    if (-dDotOC + disc > 0)
    {
        *result = ray.o + (-dDotOC + disc) * ray.d;
        return true;
    }

    return false;
}

__global__  void raytrace(unsigned char * out, int width, int height, float vFov)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Scale x and y to a range to get the desired vertical field of view angle
    // and keep the aspect ratio.
    float maxY = 1.0 / tan(vFov/2);

    float x = ((float(i % width) * 2.0 / width) - 1.0) / maxY;
    float y = ((float(i / width) * 2.0 / height) - 1.0) / maxY;

    Ray r;
    r.o = make_float3(0);
    r.d = normalize(make_float3(x, y, 1.0));

    float3 res = make_float3(0);
    if (rayIntersectSphere(r, make_float3(0, 2.0, 10.0), .5, &res))
    {
        out[4 * i]     = 255;
        out[4 * i + 1] = 0;
        out[4 * i + 2] = 0;
        out[4 * i + 3] = 255;
    }
    else
    {
        out[4 * i]     = 0;
        out[4 * i + 1] = 0;
        out[4 * i + 2] = 0;
        out[4 * i + 3] = 255;
    }
}
