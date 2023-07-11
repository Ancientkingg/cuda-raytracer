#pragma once

#include <glm/glm.hpp>
#include <curand_kernel.h>

struct HitRecord;

#include "Ray.h"
#include "Hittable.h"
#include "Texture.h"

#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ glm::vec3 random_in_unit_sphere(curandState* local_rand_state) {
	glm::vec3 p;
	do {
		p = 2.0f * RANDVEC3 - glm::vec3(1, 1, 1);
	} while (glm::dot(p, p) >= 1.0f);
	return p;
}

__device__ glm::vec3 random_in_hemisphere(curandState* local_rand_state, const glm::vec3 normal) {
	glm::vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);

	if (glm::dot(in_unit_sphere, normal) > 0.0) {
		return in_unit_sphere;
	}
	return -in_unit_sphere;
}

__device__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
	return v - 2.0f * glm::dot(v, n) * n;
}

__device__ bool near_zero(const glm::vec3& v) {
	float theta = 1e-8;
	return (fabs(v[0]) < theta) && (fabs(v[1]) < theta) && (fabs(v[2]) < theta);
}

__device__ glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
	float cos_theta = fminf(glm::dot(-uv, n), 1.0f);
	glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	glm::vec3 r_out_parallel = -sqrtf(fabsf(1.0 - glm::dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

class Material {
public:
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material {
	Texture* albedo;
public:
	__device__ Lambertian(const glm::vec3& a): albedo(new SolidColor(a)) {}
	__device__ Lambertian(Texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		glm::vec3 scatter_direction = random_in_hemisphere(local_rand_state, rec.normal);

		if (near_zero(scatter_direction)) {
			scatter_direction = rec.normal;
		}

		scattered = Ray(rec.p, scatter_direction);
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	__device__ ~Lambertian() {
		delete albedo;
	}
};

class Metal : public Material {
	glm::vec3 albedo;
	float fuzz;
public:
	__device__ Metal(const glm::vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		glm::vec3 reflected = reflect(glm::normalize(r_in.direction), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (glm::dot(scattered.direction, rec.normal) > 0.0f);
	}
};

class Dielectric : public Material {
public:
	float ir; // index of refraction

	__device__ Dielectric(float index_of_refraction): ir(index_of_refraction) {}

	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
		glm::vec3 unit_direction = glm::normalize(r_in.direction);

		float cos_theta = fminf(glm::dot(-unit_direction, rec.normal), 1.0);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;

		glm::vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = Ray(rec.p, direction);
		return true;
	}
private:
	__device__ static float reflectance(float cosine, float ref_idx) {
		// Use Schlick's approximation for reflectance.
		float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * powf((1.0f - cosine), 5);
	}
};