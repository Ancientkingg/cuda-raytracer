#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "Hittable.h"

#include <thrust/device_ptr.h>

struct kernelInfo {
    size_t buffer_size;

    thrust::device_ptr<Camera*> d_camera;
    curandState* d_rand_state;

    Hittable** d_list;
    unsigned int list_size;
    Hittable** d_world;

    cudaGraphicsResource_t resources;
    CameraInfo camera_info;

    kernelInfo() {}
    ~kernelInfo();
    kernelInfo(cudaGraphicsResource_t resources, int nx, int ny);
    void setCamera(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void render(int nx, int ny);
};