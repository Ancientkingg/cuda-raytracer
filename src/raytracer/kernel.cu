
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

#include <thrust/device_new.h>
#include <thrust/device_free.h>

#include "raytracer/kernel.h"

__global__ void raytrace(frameBuffer fb, thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera, thrust::device_ptr<curandState> rand_state) {

	// X AND Y coordinates
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// return early if we're outside of the frame buffer
	if ((i >= fb.width) || (j >= fb.height)) return;

	int pixel_idx = j * fb.width + i;

	curandState local_rand_state = rand_state[pixel_idx];

	glm::vec3 col = glm::vec3(0.0f, 0.0f, 0.0f);

	// samples
	const int ns = 3;

	for (int s = 0; s < ns; s++) {
		// normalized screen coordinates
		float u = float(i + curand_uniform(&local_rand_state)) / float(fb.width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(fb.height);
		Ray r = ((Camera*)(*d_camera))->get_ray(u, v);
		col += fb.color(r, *world, &local_rand_state);
	}
	rand_state[pixel_idx] = local_rand_state;
	col /= float(ns);
	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);

	fb.writePixel(i, j, glm::vec4(col, 1.0f));
}

__global__ void create_world(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera, CameraInfo camera_info) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new World();
		World* device_world = *d_world;

		device_world->add(new Sphere(glm::vec3(0, 0, -1), 0.5f, new Lambertian(glm::vec3(0.8f, 0.3f, 0.3f))));
		device_world->add(new Sphere(glm::vec3(0, -100.5, -1), 100.0f, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f))));
		device_world->add(new Sphere(glm::vec3(-1.01, 0, -1), 0.5f, new Dielectric(1.5f)));
		device_world->add(new Sphere(glm::vec3(-1, 10, -1), 0.5f, new Dielectric(1.5f)));
		device_world->add(new Sphere(glm::vec3(1, 0, -1), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f)));

		*d_camera = camera_info.construct_camera();
	}
}

__global__ void render_init(int width, int height, thrust::device_ptr<curandState> rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state.get()[pixel_index]);
}


kernelInfo::kernelInfo(cudaGraphicsResource_t resources, int nx, int ny) {
	this->resources = resources;
	this->nx = nx;
	this->ny = ny;

	camera_info = CameraInfo(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 90.0f, (float) nx, (float) ny);

	//checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<curandState>(nx * ny);

	d_world = thrust::device_new<World*>();

	create_world<<<1, 1>>> (d_world, d_camera, camera_info);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void kernelInfo::resize(int nx, int ny) {
	this->nx = nx;
	this->ny = ny;

	int tx = 8;
	int ty = 8;

	thrust::device_free(d_rand_state);
	d_rand_state = thrust::device_new<curandState>(nx * ny);


	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void set_camera(thrust::device_ptr<Camera*> d_camera, glm::vec3 position, glm::vec3 forward, glm::vec3 up, float aspect_ratio) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		((Camera*) (*d_camera))->set_position(position);
		((Camera*) (*d_camera))->set_rotation(forward, up, aspect_ratio);
	}
}

void kernelInfo::setCamera(glm::vec3 position, glm::vec3 forward, glm::vec3 up) {
	set_camera<<<1, 1>>> (d_camera, position, forward, up, (float) nx / (float) ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void kernelInfo::render() {
	frameBuffer fb(nx, ny);
	size_t buffer_size;

	checkCudaErrors(cudaGraphicsMapResources(1, &resources, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&fb.device_ptr, &buffer_size, resources));

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	raytrace<<<blocks, threads>>> (fb, d_world, d_camera, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	// wait for the gpu to finish
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGraphicsUnmapResources(1, &resources, NULL));
}

__global__ void free_scene(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera) {
	delete* d_world;
	delete* d_camera;
}

kernelInfo::~kernelInfo() {
	checkCudaErrors(cudaDeviceSynchronize());

	free_scene<<<1, 1>>> (d_world, d_camera);

	thrust::device_free(d_world);
	thrust::device_free(d_camera);
	thrust::device_free(d_rand_state);
}