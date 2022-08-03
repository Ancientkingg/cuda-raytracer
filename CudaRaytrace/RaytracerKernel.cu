
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
		d_list[0] = new Sphere(glm::vec3(0, 0, -1), 0.5f, new Lambertian(glm::vec3(0.8f,0.3f,0.3f)));
		d_list[1] = new Sphere(glm::vec3(0, -100.5, -1), 100.0f, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f)));
		d_list[2] = new Sphere(glm::vec3(-1, 0, -1), 0.5f, new Dielectric(1.5f));
		d_list[3] = new Sphere(glm::vec3(-1, 0, -1), -0.4f, new Dielectric(1.5f));
		d_list[4] = new Sphere(glm::vec3(1, 0, -1), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f));
		*d_world = new World(d_list, 5);
		*d_camera = new Camera();
	}
}

//__global__ void set_camera(Camera** d_camera, glm::vec3 position, glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 lower_left_corner) {
//	if (threadIdx.x == 0 && blockIdx.x == 0) {
//		(*d_camera)->setPosition(position);
//		(*d_camera)->setLookat(horizontal, vertical, lower_left_corner);
//	}
//}

__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
	delete d_list[0];
	delete d_list[1];
	delete d_list[2];
	delete d_list[3];
	delete d_list[4];
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

	// samples
	int ns = 10;

	for (int s = 0; s < ns; s++) {
		// normalized screen coordinates
		float u = float(i + curand_uniform(&local_rand_state)) / float(fb.width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(fb.height);
		Ray r = (*camera)->getRay(u, v);
		col += fb.color(r, world, &local_rand_state);
	}
	rand_state[pixel_idx] = local_rand_state;
	col /= float(ns);
	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);

	fb.writePixel(i, j, glm::vec4(col, 1.0f));
}

kernelInfo::kernelInfo(cudaGraphicsResource_t resources, int nx, int ny) {
	this->resources = resources;

	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

	checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx * ny * sizeof(curandState)));

	checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(Hittable*)));

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

//void kernelInfo::setCamera(glm::vec3 pos, glm::vec3 rotation) {
//	set_camera<<<1, 1 >>>(d_camera, pos);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//}

void kernelInfo::destroy() {

	free_world<<<1, 1>>> (d_list, d_world, d_camera);

	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
}
