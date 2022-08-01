#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"

class Sphere : public Hittable {
public:
	glm::vec3 center;
	float radius;
	__device__ Sphere() {}
	__device__ Sphere(glm::vec3 center, float radius) : center(center), radius(radius) {}
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
	glm::vec3 oc = r.origin - this->center;
	float a = glm::dot(r.direction, r.direction);
	float b = glm::dot(oc, r.direction);
	float c = glm::dot(oc, oc) - this->radius * this->radius;

	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp > t_min && temp < t_max) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}

		temp = (-b + sqrt(discriminant)) / a;
		if (temp > t_min && temp < t_max) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
	}
	return false;

}