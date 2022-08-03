#pragma once

#include "Ray.h"
#include <glm/glm.hpp>

#define M_PI   3.14159265358979323846264338327950288


__device__ inline float degrees_to_radians(float angleInDegrees) {
	return ((angleInDegrees)*M_PI / 180.0);
}

class Camera {
public:
	glm::vec3 origin;
	glm::vec3 lower_left_corner;
	glm::vec3 horizontal;
	glm::vec3 vertical;

	__device__ Camera(glm::vec3 origin, glm::vec3 forward, glm::vec3 up, float vfov, float aspect_ratio) {
		float theta = degrees_to_radians(vfov);
		float h = tan(theta / 2.0f);
		float viewport_height = 2.0f * h;
		float viewport_width = aspect_ratio * viewport_height;

		glm::vec3 w = forward;
		glm::vec3 v = up;
		glm::vec3 u = glm::cross(v, w);

		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		origin = origin;
		lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - w;
	}
	__device__ Ray getRay(float u, float v) { return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }
	/*__device__ void setPosition(glm::vec3 pos) {
		origin = pos;
	}
	__device__ void setLookat(glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 lower_left_corner) {
		horizontal = horizontal;
		vertical = vertical;
		lower_left_corner = lower_left_corner;
	}*/
};

struct CameraInfo { // used to pass all necessary information to the gpu to construct a camera on the device
	glm::vec3 origin;
	glm::vec3 rotation;
	float fov;
	float width;
	float height;

	__host__ CameraInfo() {}

	__host__ CameraInfo(glm::vec3 o, glm::vec3 r, float f, float w, float h) : origin(o), rotation(r), fov(f), width(w), height(h) {}

	__device__ Camera* constructCamera() const {
		float A = degrees_to_radians(rotation.x);
		float B = degrees_to_radians(rotation.y);
		float C = degrees_to_radians(rotation.z);

		glm::vec3 forward;
		glm::vec3 up;

		if (rotation.x == 0 && rotation.y == 0 && rotation.z == 0) {
			forward = glm::vec3(0.0f, 0.0f, 1.0f);
			up = glm::vec3(0.0f, 1.0f, 0.0f);
		}
		else {
			forward = glm::vec3(-cos(A) * sin(B) * cos(C) + sin(A) * sin(C), cos(A) * sin(B) * sin(C) + sin(A) * cos(C), cos(A) * cos(B));
			up = glm::vec3(sin(A) * sin(B) * cos(C) + cos(A) * sin(C), -sin(A) * sin(B) * sin(C) + cos(A) * cos(C), -sin(A) * cos(B));
		}


		return new Camera(origin, forward, up, fov, width / height);
	}
};