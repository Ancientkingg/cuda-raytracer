#pragma once

#include "raytracer/Camera.h"
#include "raytracer/kernel.h"
#include <glm/vec2.hpp>
#include <GLFW/glfw3.h>

struct Input {
	double last_xpos;
	double last_ypos;
	glm::vec3 speed;
	bool has_moved;

	Input() {
		speed = glm::vec3(0.0f, 0.0f, 0.0f);
		last_xpos = 0;
		last_ypos = 0;
		has_moved = false;
	}
	void processQuit(GLFWwindow* window);
	void processCameraMovement(GLFWwindow* window, kernelInfo& kernelInfo, float t_diff);
	bool hasCameraMoved();
};


