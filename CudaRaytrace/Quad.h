#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "RaytracerKernel.h"

#include <vector>

class Quad {
public:
	unsigned int VAO;
	unsigned int VBO;

	unsigned int texture;
	unsigned int PBO; // pixel buffer object
	unsigned int FBO; // frame buffer
	kernelInfo renderer;

	cudaGraphicsResource_t CGR;
	cudaArray_t CA;

	std::vector<float> vertices;
	Quad(unsigned int width, unsigned int height);

	void renderKernel(unsigned int width, unsigned int height);
};
