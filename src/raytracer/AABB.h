#pragma once

#include <glm/glm.hpp>
#include "Ray.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class AABB {
public:
	glm::vec3 _min;
	glm::vec3 _max;

	__device__ AABB() {}
	__device__ AABB(const glm::vec3& a, const glm::vec3& b) {
		_min = a;
		_max = b;
	}

	__device__ glm::vec3 min() const { return _min; }

	__device__ glm::vec3 max() const { return _max; }

	__device__ inline bool hit(const Ray& r, float t_min, float t_max) const {
		for (int a = 0; a < 3; a++) {
			auto invD = 1.0f / r.direction[a];
			auto t0 = (_min[a] - r.origin[a]) * invD;
			auto t1 = (_max[a] - r.origin[a]) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;
			if (t_max <= t_min)
				return false;
		}
		return true;
	}
};