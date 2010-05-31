
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

__global__  void raytrace(unsigned char * out, int width, int height,
                          float vFov, Sphere * spheres, int numSpheres);

