#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <assert.h>
#include <math.h>

#include "raytracer.h"

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 1024

// Handle to the pixel buffer object to write to the screen
GLuint pbo = 0;

float vFov = M_PI / 3;

// Function to update the pixels with new samples from the raytracer
void updatePixels()
{
    unsigned char * d_out;

    // Map the buffer object to some pointer to pass in
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_out, pbo));

    // Set up the grid to get a thread per pixel
    int numPixelsPerBlock = 64;
    assert(WINDOW_WIDTH % 64 == 0);

    dim3 gridDim(WINDOW_WIDTH * WINDOW_HEIGHT / numPixelsPerBlock);
    dim3 blockDim(numPixelsPerBlock);

    raytrace<<<gridDim, blockDim>>>(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, vFov);

    CUT_CHECK_ERROR("Kernel execution failed");

    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
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
            printf("Vertical FOV:%f\n", vFov);
        } break;
        case '-':
        {
            vFov /= .95;
            if (vFov > M_PI - .01)
                vFov = M_PI - .01;
            printf("Vertical FOV:%f\n", vFov);
        }break;
    }

}


void cleanup()
{
    glDeleteBuffersARB(1, &pbo);
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
    glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    // Initialize the pixel buffer object
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    cutilSafeCall(cudaGLRegisterBufferObject(pbo));

    atexit(cleanup);

    glutMainLoop();
}
