#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <assert.h>
#include <math.h>

#include "raytracer_kernel.cu"
#include "cutil_math.h"


// Handle to the pixel buffer object to write to the screen
GLuint pbo = 0;

// Some parameters of the camera
float vFov = M_PI / 3;  // Vertical field of view
float camYAngle = 0; // Angle around the vertical axis
float camGroundAngle = M_PI / 4; // Angle off the ground
float4 camMat[4];

// Array of spheres to pass into the raytracer
__device__ Sphere * d_spheres;
Sphere * spheres;
Sphere * transSpheres;
int numSpheres = 9;

// Array to keep random seeds
__device__ uint2 * d_seeds;

// Number of samples
uint numSamples;

void updateCamMat()
{
    // Calculate the position of the camera. The camera fixates on the origin
    // from 10 units away
    float  camRad = 10.0;
    float3 camPos = make_float3(camRad * cos(camYAngle) * cos(camGroundAngle),
                                camRad * sin(camGroundAngle),
                                camRad * sin(camYAngle) * cos(camGroundAngle));

    // Use glu's camera implementation to construct the view matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(camPos.x, camPos.y, camPos.z, 0, 0, 0, 0, 1, 0);

    // Load up the matrix
    float camF[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, camF);

    // Transpose the matrix, as openGL stores it in column major(?) form
    camMat[0] = make_float4(camF[0], camF[4],  camF[8],  camF[12]);
    camMat[1] = make_float4(camF[1], camF[5],  camF[9],  camF[13]);
    camMat[2] = make_float4(camF[2], camF[6],  camF[10],  camF[14]);
    camMat[3] = make_float4(camF[3], camF[7],  camF[11],  camF[15]);

    glPopMatrix();
}

void updateSpheres()
{
    // Transform the spheres by the camera transform
    for (int i = 0; i < numSpheres; i++)
    {
        float4 spherePos = make_float4(spheres[i].center, 1.0);
        float tX = dot(camMat[0], spherePos);
        float tY = dot(camMat[1], spherePos);
        float tZ = -dot(camMat[2], spherePos);
        transSpheres[i] = spheres[i];
        transSpheres[i].center = make_float3(tX, tY, tZ);
    }

    // Copy the spheres over to the device
    cutilSafeCall(cudaMemcpy(d_spheres, transSpheres,
                             numSpheres * sizeof(Sphere),
                             cudaMemcpyHostToDevice));
}

// Function to update the pixels with new samples from the raytracer
void updatePixels()
{
    // Set up the number of samples in the device
    cutilSafeCall(cudaMemcpyToSymbol(d_sampleNum, &numSamples, sizeof(uint),
                                     0, cudaMemcpyHostToDevice));

    // Set up the grid to get a thread per pixel
    int numPixelsPerBlock = 64;
    assert(WINDOW_WIDTH % numPixelsPerBlock == 0);

    dim3 gridDim(WINDOW_WIDTH * WINDOW_HEIGHT / numPixelsPerBlock);
    dim3 blockDim(numPixelsPerBlock);

    // Set enough shared memory to keep the spheres
    uint bytesPerBlock = numSpheres * sizeof(Sphere);

    // Call the raytracer kernel
    float * d_out;
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_out, pbo));
    raytrace<<<gridDim, blockDim, bytesPerBlock>>>
        (d_out, WINDOW_WIDTH, WINDOW_HEIGHT, vFov, d_spheres, numSpheres,
         d_seeds);

    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
    CUT_CHECK_ERROR("Kernel execution failed");
    numSamples++;
}
 
void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
        case '=':
        {    
            vFov *= .95;
            if (vFov < .1)
                vFov = .1;

            numSamples = 0;
        } break;
        case '-':
        {
            vFov /= .95;
            if (vFov > M_PI - .01)
                vFov = M_PI - .01;

            numSamples = 0;
        }break;
        case 'q':
        {
            exit(0);
        } break;
    }
}

void keySpecial(int key, int x, int y)
{
    switch (key)
    {
        case GLUT_KEY_LEFT:
        {
            camYAngle -= .1;
            updateCamMat();
            updateSpheres();

            numSamples = 0;
        } break;

        case GLUT_KEY_RIGHT:
        {
            camYAngle += .1;
            updateCamMat();
            updateSpheres();

            numSamples = 0;
        } break;

        case GLUT_KEY_UP:
        {
            camGroundAngle += .1;

            if (camGroundAngle > M_PI / 2 - .01)
                camGroundAngle = M_PI / 2 - .01;

            updateCamMat();
            updateSpheres();

            numSamples = 0;
        } break;

        case GLUT_KEY_DOWN:
        {
            camGroundAngle -= .1;
            if (camGroundAngle < 0)
                camGroundAngle = 0;

            updateCamMat();
            updateSpheres();

            numSamples = 0;
        } break;
    }
}

void cleanup()
{
    // Delete the pixel buffer object
    glDeleteBuffersARB(1, &pbo);

    // Free up the sphere array
    free(spheres);
    cutilSafeCall(cudaFree(d_spheres));
}

void reshape(int x, int y)
{
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// display results using OpenGL (called by GLUT)
void display()
{
    // Update the pbo
    updatePixels();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
    glutPostRedisplay();
}


void idle()
{
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("CUDA raytracer");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(keySpecial);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    // Initialize the pixel buffer object
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    cutilSafeCall(cudaGLRegisterBufferObject(pbo));

    float * d_out;

    // Map the buffer object to some pointer to pass in
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_out, pbo));

    // Zero out the buffer
    cutilSafeCall(cudaMemset(d_out, 0, sizeof(GLfloat) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT));
    cutilSafeCall(cudaGLUnmapBufferObject(pbo));

    // Initialize some spheres
    spheres = (Sphere*)malloc(sizeof(Sphere) * numSpheres);
    transSpheres = (Sphere*)malloc(sizeof(Sphere) * numSpheres);

    spheres[0].center = make_float3(0, 5, 0);
    spheres[0].radius = 1.0;
    spheres[0].emissionCol  = make_float3(2.0, 2.0, 2.0);
    spheres[0].reflectance  = make_float3(1.0, 1.0, 1.0);
    spheres[0].materialType = MATERIAL_DIFFUSE;

    spheres[1].center = make_float3(1.0, 0, 2.0);
    spheres[1].radius = 1.0;
    spheres[1].emissionCol = make_float3(0.0, 0.0, 0.0);
    spheres[1].reflectance  = make_float3(.8, 0.8, .8);
    spheres[1].materialType = MATERIAL_SPECULAR;

    spheres[2].center = make_float3(-1.0, 0, 3.0);
    spheres[2].radius = 1.0;
    spheres[2].emissionCol = make_float3(0.0, .0, 0.0);
    spheres[2].reflectance  = make_float3(1.0, 0.0, 0.0);
    spheres[2].materialType = MATERIAL_DIFFUSE;

    // The "walls"
    spheres[3].center = make_float3(10000, 0, 0);
    spheres[3].radius = 9989;
    spheres[3].emissionCol = make_float3(0.0, 0, 0.0);
    spheres[3].reflectance  = make_float3(.8, .8, .8);
    spheres[3].materialType = MATERIAL_SPECULAR;


    spheres[4].center = make_float3(-10000, 0, 0);
    spheres[4].radius = 9989;
    spheres[4].emissionCol = make_float3(0.0, 0.0, 0.0);
    spheres[4].reflectance  = make_float3(1.0, 1.0, 1.0);
    spheres[4].materialType = MATERIAL_DIFFUSE;

    spheres[5].center = make_float3(0, 10000, 0);
    spheres[5].radius = 9989;
    spheres[5].emissionCol = make_float3(0.0, .0, 0.0);
    spheres[5].reflectance  = make_float3(.8, .4, .4);
    spheres[5].materialType = MATERIAL_DIFFUSE;

    spheres[6].center = make_float3(0, -10000, 0);
    spheres[6].radius = 9999;
    spheres[6].emissionCol = make_float3(0.0, 0.0, 0.0);
    spheres[6].reflectance  = make_float3(.9, 0.95, 0.95);
    spheres[6].materialType = MATERIAL_DIFFUSE;

    spheres[7].center = make_float3(0, 0, -10000);
    spheres[7].radius = 9989;
    spheres[7].emissionCol = make_float3(0.0, .0, 0.0);
    spheres[7].reflectance  = make_float3(1.0, 1.0, 1.0);
    spheres[7].materialType = MATERIAL_DIFFUSE;

    spheres[8].center = make_float3(0, 0, 10000);
    spheres[8].radius = 9989;
    spheres[8].emissionCol = make_float3(0.0, .0, 0.0);
    spheres[8].reflectance  = make_float3(0.0, 1.0, 1.0);
    spheres[8].materialType = MATERIAL_DIFFUSE;


    // Create some sphere memory on the device
    cutilSafeCall(cudaMalloc((void**)&d_spheres,  numSpheres * sizeof(Sphere)));

    // Initialize the camera
    updateCamMat();

    // Initialize the initial transformed spheres
    updateSpheres();

    // Initialize the random seeds
    uint2 * seeds = (uint2 *) malloc(sizeof(uint2) * WINDOW_WIDTH
                                                      * WINDOW_HEIGHT);

    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
    {
        seeds[i].x = rand() % 2000000 + 1;
        seeds[i].y = rand() % 2000000 + 1;
    }
    
    cutilSafeCall(cudaMalloc((void**)&d_seeds,  sizeof(uint2) * WINDOW_WIDTH
                                                              * WINDOW_HEIGHT));
    
    cutilSafeCall(cudaMemcpy(d_seeds, seeds,
                             sizeof(uint2) * WINDOW_WIDTH * WINDOW_HEIGHT,
                             cudaMemcpyHostToDevice));

    // Initialize the sample count
    numSamples = 0;

    atexit(cleanup);

    glutMainLoop();
}
