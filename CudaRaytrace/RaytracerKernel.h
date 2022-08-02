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
    Hittable** d_world;
    cudaGraphicsResource_t resources;

    kernelInfo() {}
    kernelInfo(cudaGraphicsResource_t resources, int nx, int ny);
    void render(int nx, int ny);
    void destroy();
};
