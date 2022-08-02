
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include <glm/glm.hpp>
#include <curand_kernel.h>

#include "Window.h"
#include "cuda_errors.h"
#include "FrameBuffer.h"
#include "Ray.h"
#include "Sphere.h"
#include "World.h"
#include "Camera.h"

#include "RaytracerKernel.h"

__global__ void create_world(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new Sphere(glm::vec3(0, 0, -1), 0.5f);
		d_list[1] = new Sphere(glm::vec3(0, -100.5, -1), 100.0f);
		*d_world = new World(d_list, 2);
		*d_camera = new Camera();
	}
}

__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
	delete d_list[0];
	delete d_list[1];
	delete* d_world;
	delete* d_camera;
}

__global__ void render_init(int width, int height, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void raytrace(frameBuffer fb, Hittable** world, Camera** camera, curandState* rand_state) {

	// X AND Y coordinates
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// return early if we're outside of the frame buffer
	if ((i >= fb.width) || (j >= fb.height)) return;

	int pixel_idx = j * fb.width + i;

	curandState local_rand_state = rand_state[pixel_idx];

	glm::vec3 col = glm::vec3(0.0f, 0.0f, 0.0f);

	for (int s = 0; s < 10; s++) {
		// normalized screen coordinates
		float u = float(i + curand_uniform(&local_rand_state)) / float(fb.width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(fb.height);
		Ray r = (*camera)->getRay(u, v);
		col += fb.color(r, world, &local_rand_state);
	}
	rand_state[pixel_idx] = local_rand_state;
	col /= float(10);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);

	fb.writePixel(i, j, glm::vec4(col, 1.0f));
}

kernelInfo::kernelInfo(cudaGraphicsResource_t resources, int nx, int ny) {
	this->resources = resources;

	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

	checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx * ny * sizeof(curandState)));

	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));

	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void kernelInfo::render(int nx, int ny) {
	frameBuffer fb(nx, ny);

	checkCudaErrors(cudaGraphicsMapResources(1, &resources, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&fb.device_ptr, &buffer_size, resources));

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	raytrace << <blocks, threads >> > (fb, d_world, d_camera, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	// wait for the gpu to finish
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGraphicsUnmapResources(1, &resources, NULL));
}

void kernelInfo::destroy() {

	free_world<<<1, 1>>> (d_list, d_world, d_camera);

	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
}



/*
void kernel(cudaGraphicsResource_t resources, int nx, int ny)
{
	// create framebuffer
	frameBuffer fb(nx, ny);

	// create camera;
	Camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

	// assign opengl resources to this buffer and map a pointer to it
	size_t buffer_size;
	checkCudaErrors(cudaGraphicsMapResources(1, &resources, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&fb.device_ptr, &buffer_size, resources));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx*ny * sizeof(curandState)));

	// make our world of hittables
	Hittable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));

	Hittable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
	create_world<<<1, 1>>> (d_list, d_world, d_camera);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// divide work on the gpu into blocks of 8x8 threads
	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks,threads>>>(fb, d_world, d_camera, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	// wait for the gpu to finish
	checkCudaErrors(cudaDeviceSynchronize());

	free_world <<<1, 1>>> (d_list, d_world, d_camera);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &resources, NULL));
}
*/