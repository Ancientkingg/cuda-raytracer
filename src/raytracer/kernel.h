#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "World.h"

#include <thrust/device_ptr.h>

struct KernelInfo {

    thrust::device_ptr<Camera*> d_camera;
    thrust::device_ptr<curandState> d_rand_state;
    
    thrust::device_ptr<World*> d_world;

    cudaGraphicsResource_t resources;
    CameraInfo camera_info;

    int nx, ny;

    KernelInfo() {}
    ~KernelInfo();
    KernelInfo(cudaGraphicsResource_t resources, int nx, int ny);
    void set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void render();
    void resize(int nx, int ny);
};