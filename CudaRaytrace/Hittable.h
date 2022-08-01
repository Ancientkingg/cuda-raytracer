#pragma once

#include "Ray.h"
#include <glm/glm.hpp>

struct HitRecord {
	float t;
	glm::vec3 p;
	glm::vec3 normal;
};

class Hittable {
public:
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};