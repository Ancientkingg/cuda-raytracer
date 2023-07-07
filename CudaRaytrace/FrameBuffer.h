#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include "device_launch_parameters.h"
#include "Hittable.h"
#include <curand_kernel.h>
#include "Material.h"

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

	__device__ glm::vec3 color(const Ray& r, Hittable** world, curandState* local_rand_state) {
		Ray cur_ray = r;
		glm::vec3 cur_attenuation = glm::vec3(1.0, 1.0, 1.0);
		// ray depth
		for (int i = 0; i < 50; i++) {
			HitRecord rec;
			if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
				Ray scattered;
				glm::vec3 attenuation;
				if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
					cur_attenuation *= attenuation;
					cur_ray = scattered;
				}
				else {
					return glm::vec3(0.0f, 0.0f, 0.0f);
				}
			}
			else {
				glm::vec3 unit_direction = glm::normalize(cur_ray.direction);
				float t = 0.5f * (unit_direction.y + 1.0f);
				glm::vec3 c = (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
				return cur_attenuation * c;
			}
		}
		return glm::vec3(0.0f, 0.0f, 0.0f);
	}

	__host__ void free() {
		checkCudaErrors(cudaFree(this->device_ptr));
		checkCudaErrors(cudaFree(this));
	}
};