#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"

#define M_PI 3.14159265358979323846264338327950288f

class Sphere : public Hittable {
public:
	glm::vec3 center;
	float radius;
	Material* mat_ptr;

	__device__ Sphere() {}
	__device__ ~Sphere() {
		delete mat_ptr;
	}

	__device__ Sphere(glm::vec3 center, float radius, Material* mat_ptr) : center(center), radius(radius), mat_ptr(mat_ptr) {}
	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
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
				glm::vec3 outward_normal = (rec.p - center) / radius;
				rec.set_face_normal(r, outward_normal);
				get_sphere_uv(outward_normal, rec.u, rec.v);
				rec.mat_ptr = mat_ptr;
				return true;
			}

			temp = (-b + sqrt(discriminant)) / a;
			if (temp > t_min && temp < t_max) {
				rec.t = temp;
				rec.p = r.at(rec.t);
				glm::vec3 outward_normal = (rec.p - center) / radius;
				rec.set_face_normal(r, outward_normal);
				get_sphere_uv(outward_normal, rec.u, rec.v);
				rec.mat_ptr = mat_ptr;
				return true;
			}
		}
		return false;

	}

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const {
		output_box = AABB(center - glm::vec3(radius, radius, radius), center + glm::vec3(radius, radius, radius));
		return true;
	}

	__device__ static void get_sphere_uv(const glm::vec3& p, float& u, float& v) {
		float theta = acosf(-p.y);
		float phi = atan2f(-p.z, p.x) + M_PI;

		u = phi / (2 * M_PI);
		v = theta / M_PI;
	}
};