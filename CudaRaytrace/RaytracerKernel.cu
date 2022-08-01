
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include <glm/glm.hpp>

#include "Window.h"
#include "cuda_errors.h"
#include "FrameBuffer.h"
#include "Ray.h"
#include "Sphere.h"
#include "World.h"

#include "RaytracerKernel.h"

__global__ void create_world(Hittable** d_list, Hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(glm::vec3(0, 0, -1), 0.5f);
        d_list[1] = new Sphere(glm::vec3(0, -100.5, -1), 100.0f);
        *d_world = new World(d_list, 2);
    }
}

__global__ void free_world(Hittable** d_list, Hittable** d_world) {
    delete d_list[0];
    delete d_list[1];
    delete* d_world;
}

__global__ void render(frameBuffer fb, Hittable** world) {

    // X AND Y coordinates
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // return early if we're outside of the frame buffer
    if ((i >= fb.width) || (j >= fb.height)) return;


    // normalized screen coordinates
    float u = float(i) / float(fb.width);
    float v = float(j) / float(fb.height);

    const float aspect_ratio = float(fb.width) / fb.height;

    // Camera

    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    glm::vec3 origin = glm::vec3(0, 0, 0);
    glm::vec3 horizontal = glm::vec3(viewport_width, 0, 0);
    glm::vec3 vertical = glm::vec3(0, viewport_height, 0);
    glm::vec3 lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - glm::vec3(0, 0, focal_length);

    Ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);

    fb.writePixel(i, j, glm::vec4(fb.color(r, world),1.0));
}

void kernel(cudaGraphicsResource_t resources, int nx, int ny)
{
    // create framebuffer
    frameBuffer fb(nx, ny);

    // assign opengl resources to this buffer and map a pointer to it
    size_t buffer_size;
    checkCudaErrors(cudaGraphicsMapResources(1, &resources, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&fb.device_ptr, &buffer_size, resources));

    // make our world of hitables
    Hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));
    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    create_world<<<1, 1>>> (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // divide work on the gpu into blocks of 8x8 threads
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks,threads>>>(fb, d_world);
    checkCudaErrors(cudaGetLastError());
    // wait for the gpu to finish
    checkCudaErrors(cudaDeviceSynchronize());

    free_world <<<1, 1>>> (d_list, d_world);
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resources, NULL));    
}

