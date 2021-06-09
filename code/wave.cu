////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

// This aligns with the NVIDIA CUDA samples at https://github.com/NVIDIA/cuda-samples.

/*
	This example demonstrates how to use the Cuda OpenGL bindings to
	dynamically modify a vertex buffer using a Cuda kernel.

	The steps are:
	1. Create an empty vertex buffer object (VBO)
	2. Register the VBO with Cuda
	3. Map the VBO for writing from Cuda
	4. Run Cuda kernel to modify the vertex positions
	5. Unmap the VBO
	6. Render the results using OpenGL

	Host code
*/

#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdlib>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <vector_types.h>

const unsigned REFRESH_DELAY = 10; //ms



////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned window_width = 512;
const unsigned window_height = 512;

const unsigned mesh_width = 256;
const unsigned mesh_height = 256;
const unsigned mesh_depth = 256;

// vbo variables
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource = nullptr;
void* d_vbo_buffer = nullptr;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface* timer = nullptr;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;

////////////////////////////////////////////////////////////////////////////////
// variables

int waveSelect;
float g_fUserAnim = 0.01f;
float meshR = 1.0f, meshG = 1.0f, meshB = 1.0f;
float userFreq = 4.0f;
int user_mesh_height = 256;
int circlePosX = mesh_width / 2;
int circlePosY = user_mesh_height / 2;
float circleRadius = 20;
int circlePosZ = (int)circleRadius * 2;

bool circleCheck = true;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool run(int argc, char** argv);
void cleanup();

// GL functionality
bool initGL(int* argc, char** argv);
void createVBO(GLuint* vbo, cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, cudaGraphicsResource* vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(cudaGraphicsResource** vbo_resource);

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4* pos, unsigned int width, unsigned int height, unsigned int depth, float time, int waveSelect, float userFreq, int circlePosX, int circlePosY, int circlePosZ, float circleRadius, bool circleCheck)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//All references to a Z coordinate were an attempt to generate a Sphere
//	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
//	float t = z / (float)depth;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
//	t = t * 2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = userFreq;
	float w[3];

	w[0] = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
//	w[0] = sinf(u * freq + time) * cosf(v * freq + time) * sinf(t * freq * time) * 0.5f;
	w[1] = cosf(u * freq + time) * sinf(v * freq + time) * 0.5f;
	w[2] = sinf(u * freq + time) * tanf(v * freq + time) * 0.5f;
	w[3] = cosf(u * freq + time) * tanf(v * freq + time) * 0.5f;

	GLfloat circleCenterX = x - ((float)circlePosX);
	GLfloat circleCenterY = y - ((float)circlePosY);
//	GLfloat circleCenterZ = z - ((float)circlePosZ);

	if (circleCheck)
	{
		if ((circleCenterX * circleCenterX) + (circleCenterY * circleCenterY) < (circleRadius * circleRadius))
		{
			w[waveSelect] = 0;
		}
		else
		{
			w[waveSelect];
		}
	}

	//if (circleCheck)
	//{
	//	if ((circleCenterX * circleCenterX) + (circleCenterY * circleCenterY) + (circleCenterZ * circleCenterZ) < (circleRadius * circleRadius))
	//	{
	//		w[waveSelect] = 0;
	//	}
	//	else
	//	{
	//		w[waveSelect];
	//	}
	//}

	//if (waveSelect == 0)
	//{
	//	w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
	//}
	//else if (waveSelect == 1)
	//{
	//	w = cosf(u * freq + time) * sinf(v * freq + time) * 0.5f;
	//}
	//else if (waveSelect == 2)
	//{
	//	w = sinf(u * freq + time) * tanf(v * freq + time) * 0.5f;
	//}
	//else if (waveSelect == 3)
	//{
	//	w = cosf(u * freq + time) * tanf(v * freq + time) * 0.5f;
	//}


	// write output vertex
	pos[y * width + x] = make_float4(u, w[waveSelect], v, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	std::cout << "wave assignment program starting...\n";

	run(argc, argv);

	std::cout << "wave assignment program completed.\n";

	return 0;
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char variables[256];
	sprintf(variables, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz) Mesh Height: %u Freq: %3.1f Speed: %3.1f R: %3.1f G: %3.1f B: %3.1f", avgFPS, user_mesh_height, userFreq, g_fUserAnim, meshR, meshG, meshB);
	glutSetWindowTitle(variables);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
		std::cerr << "ERROR: GL_ARB_pixel_buffer_object support missing.";
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	assert(GL_NO_ERROR == glGetError());
	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool run(int argc, char** argv)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int devID = findCudaDevice(argc, (const char**)argv);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource** vbo_resource)
{

	// map OpenGL buffer object for writing from CUDA
	float4* dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
		*vbo_resource));
	//std::cout << "CUDA mapped VBO: May access " << num_bytes << " bytes\n";

	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << <grid, block >> > (dptr, mesh_width, user_mesh_height, mesh_depth, g_fAnim, waveSelect, userFreq, circlePosX, circlePosY, circlePosZ, circleRadius, circleCheck);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	assert(GL_NO_ERROR == glGetError());
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(meshR, meshG, meshB);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	g_fAnim += g_fUserAnim;

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
		glutDestroyWindow(glutGetWindow());
	case ('.'):
		if (g_fUserAnim < 0.10f)
			g_fUserAnim += 0.01f;
		break;
	case (','):
		if (g_fUserAnim >= 0.01f)
			g_fUserAnim -= 0.01f;
		if (g_fUserAnim < 0.01f)
			g_fUserAnim = 0.01f;
		break;
	case ('p'):
		if (meshR < 1.0f)
			meshR += 0.1f;
		break;
	case ('o'):
		if (meshR >= 0.0f)
			meshR -= 0.1f;
		break;
	case ('l'):
		if (meshG < 1.0f)
			meshG += 0.1f;
		break;
	case ('k'):
		if (meshG >= 0.0f)
			meshG -= 0.1f;
		break;
	case ('m'):
		if (meshB < 1.0f)
			meshB += 0.1f;
		break;
	case ('n'):
		if (meshB >= 0.0f)
			meshB -= 0.1f;
		break;
	case ('x'):
		if (userFreq < 10.0f)
			userFreq += 1.0f;
		break;
	case ('z'):
		if (userFreq >= 1.0f)
			userFreq -= 1.0f;
		if (userFreq < 1.0f)
			userFreq = 1.0f;
		break;
	case ('c'):
		if (user_mesh_height >= 1)
			user_mesh_height -= 1;
		if (user_mesh_height < 1)
			user_mesh_height = 1;
		break;
	case ('v'):
		if (user_mesh_height < 1024)
			user_mesh_height += 1;
		break;
	case ('w'):
		circlePosY++;
		break;
	case ('s'):
		circlePosY--;
		break;
	case ('d'):
		circlePosX++;
		break;
	case ('a'):
		circlePosX--;
		break;
	case ('1'):
		waveSelect = 0;
		break;
	case ('2'):
		waveSelect = 1;
		break;
	case ('3'):
		waveSelect = 2;
		break;
	case ('4'):
		waveSelect = 3;
		break;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}