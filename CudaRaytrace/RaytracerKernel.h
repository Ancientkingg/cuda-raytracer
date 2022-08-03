#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "Hittable.h"

extern void kernel(cudaGraphicsResource_t resources, int nx, int ny);

struct kernelInfo {
    size_t buffer_size;
    Camera** d_camera;
    curandState* d_rand_state;
    Hittable** d_list;
    unsigned int list_size;
    Hittable** d_world;
    cudaGraphicsResource_t resources;
    CameraInfo camera_info;

    kernelInfo() {}
    kernelInfo(cudaGraphicsResource_t resources, int nx, int ny);
    void setCamera(glm::vec3 speed, glm::vec3 rotation);
    void render(int nx, int ny);
    void destroy();
};
