
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

__global__ void raytrace(FrameBuffer fb, thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera, thrust::device_ptr<curandState> rand_state) {

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


KernelInfo::KernelInfo(cudaGraphicsResource_t resources, int nx, int ny) {
	this->resources = resources;
	this->nx = nx;
	this->ny = ny;

	this->frame_buffer = new FrameBuffer(nx, ny);

	camera_info = CameraInfo(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 90.0f, (float) nx, (float) ny);

	//checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<curandState>(nx * ny);

	d_world = thrust::device_new<World*>();

	create_world<<<1, 1>>> (d_world, d_camera, camera_info);

	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

void KernelInfo::resize(int nx, int ny) {
	this->nx = nx;
	this->ny = ny;

	delete frame_buffer;
	this->frame_buffer = new FrameBuffer(nx, ny);

	int tx = 8;
	int ty = 8;

	thrust::device_free(d_rand_state);
	d_rand_state = thrust::device_new<curandState>(nx * ny);


	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

__global__ void set_device_camera(thrust::device_ptr<Camera*> d_camera, glm::vec3 position, glm::vec3 forward, glm::vec3 up, float aspect_ratio) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		((Camera*) (*d_camera))->set_position(position);
		((Camera*) (*d_camera))->set_rotation(forward, up, aspect_ratio);
	}
}

void KernelInfo::set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up) {
	set_device_camera<<<1, 1>>> (d_camera, position, forward, up, (float) nx / (float) ny);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

void KernelInfo::render() {

	check_cuda_errors(cudaGraphicsMapResources(1, &resources));
	check_cuda_errors(cudaGraphicsResourceGetMappedPointer((void**)&(frame_buffer->device_ptr), &(frame_buffer->buffer_size), resources));

	int tx = 32;
	int ty = 32;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	// frame buffer is implicitly copied to the device each frame
	raytrace<<<blocks, threads>>> (*frame_buffer, d_world, d_camera, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	// wait for the gpu to finish
	check_cuda_errors(cudaDeviceSynchronize());

	check_cuda_errors(cudaGraphicsUnmapResources(1, &resources));
}

__global__ void free_scene(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera) {
	delete* d_world;
	delete* d_camera;
}

KernelInfo::~KernelInfo() {
	check_cuda_errors(cudaDeviceSynchronize());

	free_scene<<<1, 1>>> (d_world, d_camera);

	thrust::device_free(d_world);
	thrust::device_free(d_camera);
	thrust::device_free(d_rand_state);

	delete frame_buffer;
}