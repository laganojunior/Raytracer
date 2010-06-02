#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 1024

#include "cutil_math.h"

// Inifinity definition
#define CUDA_INF __int_as_float(0x7f800000)


// types of materials
#define MATERIAL_DIFFUSE 1  // A perfectly diffuse material
#define MATERIAL_SPECULAR 2 // A perfectly specular material

struct __align__(16) Sphere
{
    float3 center;
    float  radius;
    
    // Emission values, assume that the sphere emits diffusely at all points
    float3 emissionCol;

    int materialType;

    // Proportion of light reflected when light approaches the point along
    // the normal. For perfectly diffuse objects, this value represents the
    // reflectance along all directions.
    float3 reflectance;

};

struct __align__(16) Ray
{
    float3 o;
    float3 d;
};

struct __align__(16) EyePathNode
{
    float3 emission;
    float3 reflectance;
    int reflectOff; // Material type hit
};

// The maximum depth to go up to. This is hardcoded in compile time.
#define MAX_DEPTH 20

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
__device__ int nearestSphere(Ray r, Sphere * spheres, int numSpheres,
                             float3 * intersectP)
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
            *intersectP = r.o + dist * r.d;
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
// Return a random reflection vector given a normal using cosine density
// (i.e. more likely to reflect toward the normal)
///////////////////////////////////////////////////////////////////////////////
__device__ float3 getRandRef(float3 normal, uint2 * seed)
{
    // Get a random rotation (phi) about the normal and angle off the plane
    // (theta) perpendicular to the normal
    float cosTheta = sqrt(1 - getRandFloat(seed));
    float sinTheta = sqrt(1 - cosTheta * cosTheta);
    float phi = 2.0 * 3.14 * getRandFloat(seed);

    // Construct some uvw basis respective to the normal, such that the
    // w axis goes along the normal.
    float3 w = normal;

    // initial value for v can be arbitrary as long as its not parallel
    // to the normal
    float3 v = normalize(normal + make_float3(normal.z, normal.x, normal.y));
    
    // calculate v and u using cross products, note that sign doesn't really
    // matter
    float3 u = cross(w, v);
    v = cross(w, u);

    // Use the basis and the angles to calculate the reflection
    float3 ref = u * (sinTheta * cos(phi))
               + v * (sinTheta * sin(phi))
               + w * (cosTheta);

    return normalize(ref);
}

///////////////////////////////////////////////////////////////////////////////
// Return the perfectly reflected vector
///////////////////////////////////////////////////////////////////////////////
__device__ float3 getRefPerfect(float3 n, float3 in)
{
    return in - 2.0 * dot(in, n) * n;
}

///////////////////////////////////////////////////////////////////////////////
// Return the sphere normal at some point
///////////////////////////////////////////////////////////////////////////////
__device__ float3 getNormal(Sphere s, float3 p)
{
    return normalize(p - s.center);
}

///////////////////////////////////////////////////////////////////////////////
// Build a path, collecting light samples along the way. The depth traveled
// is stored in depth.
///////////////////////////////////////////////////////////////////////////////
__device__ void buildPath(int i, Ray r, Sphere * spheres, int numSpheres,
                          uint2 * seed, int * depth, EyePathNode * pathInf)
{
    int dCount = 0; // How far the path is

    while (dCount < MAX_DEPTH)
    {
        // Get the next sphere to hit
        float3 p;
        int next = nearestSphere(r, spheres, numSpheres, &p);

        if (next == -1)
            break;

        // Store the emission of this sphere
        pathInf[dCount].emission = spheres[next].emissionCol; 

        // Check the type of material to see the next ray to shoot and the
        // reflectance value
        float3 nextDir;
        if (spheres[next].materialType == MATERIAL_DIFFUSE)
        {
            // Diffuse uses cosine density to reflect
            nextDir = getRandRef(getNormal(spheres[next], p), seed);
            
            r.o = p;
            r.d = nextDir;

            // Reflectance is constant regardless of direction
            pathInf[dCount].reflectance = spheres[next].reflectance;

            pathInf[dCount].reflectOff = MATERIAL_DIFFUSE;
        }
        else if (spheres[next].materialType == MATERIAL_SPECULAR)
        {
            // Perfectly specular surfaces perfect reflect the ray
            float3 n = getNormal(spheres[next], p);
            nextDir = getRefPerfect(n, r.d);
            
            r.o = p;
            r.d = nextDir;

            // Reflectance is modified depending on how the ray hits
            // the material

            // One minus cos (theta), theta is the angle the reflection
            // makes off the normal
            float omct = 1 - dot(nextDir, n);
            float omct5 = omct * omct * omct * omct * omct;
            
            pathInf[dCount].reflectance = spheres[next].reflectance
                                    + (make_float3(1.0, 1.0, 1.0)
                                       - spheres[next].reflectance) * omct5; 

            pathInf[dCount].reflectOff = MATERIAL_SPECULAR;
        }

        dCount ++;
    }

    *depth = dCount;
}

///////////////////////////////////////////////////////////////////////////////
// Combine the path information to get a sample
//////////////////////////////////////////////////////////////////////////////
__device__ float3 getSample(int i, int depth, EyePathNode * pathInf)
{
    float3 sample = pathInf[depth - 1].emission;
    depth--;
    while (depth > 0)
    {
        sample = pathInf[depth-1].emission
               + pathInf[depth-1].reflectance * sample;
        depth --;
    }

    return sample;
}

///////////////////////////////////////////////////////////////////////////////
// The kernel function to handle one pass of the raytracer
///////////////////////////////////////////////////////////////////////////////
__global__  void raytrace(unsigned char * out, int width, int height,
                          float vFov, Sphere * spheres, int numSpheres,
                          uint2 * seeds)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    extern __shared__ Sphere s_spheres[];

    // Load up spheres into shared memory
    int startI = 0;
    while (startI < numSpheres)
    {
        if (startI + threadIdx.x < numSpheres)
           s_spheres[startI + threadIdx.x] = spheres[startI + threadIdx.x];

        startI += blockDim.x;
    }
  
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

    // Recursively build up a path and store necessary light information
    EyePathNode pathInf[MAX_DEPTH];
    int depth;
    buildPath(i, r, s_spheres, numSpheres, &seed, &depth, pathInf); 

    float3 sample = getSample(i, depth, pathInf);
    addSample(out, i, make_uint4(min(sample.x * 255, 255.0),
                                 min(sample.y * 255, 255.0),
                                 min(sample.z * 255, 255.0),
                                 255));
    // Write the seed back
    seeds[i] = seed;
}
