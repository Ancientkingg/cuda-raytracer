#pragma once

#include "Ray.h"
#include <glm/glm.hpp>


class Camera {
public:
	glm::vec3 origin;
	glm::vec3 lower_left_corner;
	glm::vec3 horizontal;
	glm::vec3 vertical;
	__device__ Camera() {
		float aspect_ratio = 16.0f / 9.0f;
		float viewport_height = 2.0f;
		float viewport_width = aspect_ratio * viewport_height;
		float focal_length = 1.0f;

		
		horizontal = glm::vec3(viewport_width, 0.0f, 0.0f);
		vertical = glm::vec3(0.0f, viewport_height, 0.0f);
		origin = glm::vec3(0.0f, 0.0f, 0.0f);
		lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - glm::vec3(0, 0, focal_length);
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