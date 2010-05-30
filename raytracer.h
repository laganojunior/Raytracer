struct Ray
{
    float3 o;
    float3 d; 
};

__global__  void raytrace(unsigned char * out, int width, int height, float vFov);

