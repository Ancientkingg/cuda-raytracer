
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "Window.h"
#include "cuda_errors.h"

#include "kernel.h"

__global__ void render(uint8_t* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 4 + i * 4;
    fb[pixel_index + 0] = (uint8_t) (float(i) / max_x * 255);
    fb[pixel_index + 1] = (uint8_t) (float(j) / max_y * 255);
    fb[pixel_index + 2] = (uint8_t) (0.2 * 255);
    fb[pixel_index + 3] = (uint8_t) (0.2 * 255);
}

void kernel(cudaGraphicsResource_t resources, int nx, int ny)
{

    // pointer to buffer and allocation of it
    uint8_t* device_ptr;
    size_t buffer_size;
    checkCudaErrors(cudaGraphicsMapResources(1, &resources, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &buffer_size, resources));

    // divide work on the gpu into blocks of 8x8 threads
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks,threads>>>(device_ptr, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsUnmapResources(1, &resources, NULL));    
}

