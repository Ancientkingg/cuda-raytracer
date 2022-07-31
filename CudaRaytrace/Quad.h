#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <vector>

class Quad {
public:
	unsigned int VAO;
	unsigned int VBO;

	unsigned int texture;
	unsigned int PBO; // pixel buffer object
	unsigned int RBO; // depth buffer
	unsigned int FBO; // frame buffer
	
	cudaGraphicsResource_t CGR;
	cudaArray_t CA;

	std::vector<float> vertices;
	Quad(unsigned int width, unsigned int height);
};
