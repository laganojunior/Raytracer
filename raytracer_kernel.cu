#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 1024

#include "cutil_math.h"

// Inifinity definition
#define CUDA_INF __int_as_float(0x7f800000)

struct __align__(16) Sphere
{
    float3 center;
    float  radius;
    
    // Emission values, assume that the sphere emits diffusely at all points
    float3 emissionCol;

};

struct Ray
{
    float3 o;
    float3 d;
};

// Number of samples taken so far
__constant__ uint d_sampleNum;

__global__  void raytrace(unsigned char * out, int width, int height,
                          float vFov, Sphere * spheres, int numSpheres,
                          uint2 * seeds);


/////////////////////////////////////////////////////////////////////////////
// A simple multiply with carry 32 bit random number generator. The state
// of the random number generator is carried in the arguments. Taken
// from the Wikipedia page on random number generation.
/////////////////////////////////////////////////////////////////////////////
__device__ unsigned int getRand(uint2 * seed)
{
    seed->x = 36969 * (seed->x & 0xFFFF) + (seed->y >> 16);
    seed->y = 18000 * (seed->y & 0xFFFF) + (seed->x >> 16);

    return (seed->y << 16) + seed->x; 
}

/////////////////////////////////////////////////////////////////////////////
// Return a random float from 0 to 1
/////////////////////////////////////////////////////////////////////////////
__device__ float getRandFloat(uint2 * seed)
{
    return ((float)getRand(seed)) / 4294967296.0;
}

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
// Average in the current sample
///////////////////////////////////////////////////////////////////////////////
__device__ void addSample(unsigned char * out, int i, uint4 sample)
{
    
    out[4 * i]   = (out[4 * i]  * d_sampleNum + sample.x)   / (d_sampleNum + 1);
    out[4 * i+1] = (out[4 * i+1] * d_sampleNum + sample.y) / (d_sampleNum + 1);
    out[4 * i+2] = (out[4 * i+2] * d_sampleNum + sample.z) / (d_sampleNum + 1);
    out[4 * i+3] = (out[4 * i+3] * d_sampleNum + sample.w) / (d_sampleNum + 1);
    
}

///////////////////////////////////////////////////////////////////////////////
// The kernel function to handle one pass of the raytracer
///////////////////////////////////////////////////////////////////////////////
__global__  void raytrace(unsigned char * out, int width, int height,
                          float vFov, Sphere * spheres, int numSpheres,
                          uint2 * seeds)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        
    // Get the random seed for this pixel
    uint2 seed = seeds[i];

    // Scale x and y to a range to get the desired vertical field of view angle
    // and keep the aspect ratio.
    float maxY = 1.0 / tan(vFov/2);

    float x = ((float(i % width) * 2.0 / width) - 1.0) / maxY;
    float y = ((float(i / width) * 2.0 / height) - 1.0) / maxY;

    // Jiggle the ray a bit to get natural anti-aliasing
    x += (getRandFloat(&seed) * 2.0 - 1.0) / width / maxY;
    y += (getRandFloat(&seed) * 2.0 - 1.0) / height / maxY;

    // Generate the initial ray for this pixel
    Ray r;
    r.o = make_float3(0);
    r.d = normalize(make_float3(x, y, 1.0));

    int nearest = nearestSphere(r, spheres, numSpheres);

    if (nearest != -1)
    {
        addSample(out, i, make_uint4(
                            min(spheres[nearest].emissionCol.x * 255, 255.0),
                            min(spheres[nearest].emissionCol.y * 255, 255.0),
                            min(spheres[nearest].emissionCol.z * 255, 255.0),
                            255));
    }
    else
    {
        addSample(out, i, make_uint4(0, 0, 0, 255));
    }

    // Write the seed back
    seeds[i] = seed;
}
