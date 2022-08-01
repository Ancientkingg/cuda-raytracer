#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include "device_launch_parameters.h"
#include "Hittable.h"
#include <curand_kernel.h>

class frameBuffer {
public:
	uint32_t* device_ptr; // RGBA8 internal format, but uses BGRA
	unsigned int width;
	unsigned int height;


	__host__ frameBuffer(unsigned int width, unsigned int height) : width{ width }, height{ height } {}

	__device__ void writePixel(int x, int y, glm::vec4 pixel) {
		int idx = y * width + x;
		// convert RGBA to BGRA that buffer uses
		device_ptr[idx] = glm::packUnorm4x8(glm::vec4(pixel.b, pixel.g, pixel.r, pixel.a));
	}

	__device__ glm::vec3 color(const Ray& r, Hittable** world) {
		HitRecord rec;

		if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
			return 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
		}

		glm::vec3 unit_direction = glm::normalize(r.direction);
		float t = 0.5f * (unit_direction.y + 1.0f);
		return (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
	}

	__host__ void free() {
		checkCudaErrors(cudaFree(this->device_ptr));
		checkCudaErrors(cudaFree(this));
	}
};