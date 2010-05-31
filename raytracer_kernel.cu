#include "raytracer.h"
#include "cutil_math.h"

/////////////////////////////////////////////////////////////////////////////
// Function to calculate the intersection point of a ray with a sphere. 
// Assumes the ray has unit length
////////////////////////////////////////////////////////////////////////////
__device__ float rayIntersectSphere(Ray ray, float3 center, float radius)
{
    float3 oMinusC = ray.o - center; 
    float  dDotOC  = dot(ray.d, oMinusC);
    float disc = dDotOC * dDotOC - (dot(oMinusC, oMinusC) - radius * radius);

    if (disc < 0)
        return CUDA_INF;

    disc = sqrt(disc);

    if (-dDotOC - disc > 0)
        return -dDotOC - disc;

    if (-dDotOC + disc > 0)
        return -dDotOC + disc;

    return CUDA_INF;
}


///////////////////////////////////////////////////////////////////////////////
// Return the nearest sphere index. If none intesect, -1 is returned
///////////////////////////////////////////////////////////////////////////////
__device__ int nearestSphere(Ray r, Sphere * spheres, int numSpheres)
{
    int best = -1;
    float dist = CUDA_INF; 

    for (int i = 0; i < numSpheres; i++)
    {
        float thisDist = rayIntersectSphere(r, spheres[i].center,
                                            spheres[i].radius);
        if (thisDist < dist)
        {
            dist = thisDist;
            best = i;
        }
    } 

    return best;
}

///////////////////////////////////////////////////////////////////////////////
// The kernel function to handle one pass of the raytracer
///////////////////////////////////////////////////////////////////////////////
__global__  void raytrace(unsigned char * out, int width, int height,
                          float vFov, Sphere * spheres, int numSpheres)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Scale x and y to a range to get the desired vertical field of view angle
    // and keep the aspect ratio.
    float maxY = 1.0 / tan(vFov/2);

    float x = ((float(i % width) * 2.0 / width) - 1.0) / maxY;
    float y = ((float(i / width) * 2.0 / height) - 1.0) / maxY;

    // Generate the initial ray for this pixel
    Ray r;
    r.o = make_float3(0);
    r.d = normalize(make_float3(x, y, 1.0));

    int nearest = nearestSphere(r, spheres, numSpheres);

    if (nearest != -1)
    {
        out[4*i]     = min(spheres[nearest].emissionCol.x * 255, 255.0);
        out[4*i + 1] = min(spheres[nearest].emissionCol.y * 255, 255.0);
        out[4*i + 2] = min(spheres[nearest].emissionCol.z * 255, 255.0);
        out[4*i + 3] = 255;

    }
    else
    {
        out[4 * i]     = 0;
        out[4 * i + 1] = 0;
        out[4 * i + 2] = 0;
        out[4 * i + 3] = 255;

    }
}
