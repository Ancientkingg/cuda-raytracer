#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include <glm/glm.hpp>

#include "Input.h"


#define M_PI 3.14159265358979323846264338327950288f

#define isPressed(x) glfwGetKey(window,x)==GLFW_PRESS 

void Input::process_quit(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

bool Input::has_camera_moved() {
	return has_moved;
}

void Input::process_camera_movement(GLFWwindow* window, KernelInfo& kernelInfo, float t_diff) {
	has_moved = false;
	glm::vec3 position = kernelInfo.camera_info.origin;
	glm::vec3 rotation = kernelInfo.camera_info.rotation;

	// ** rotation **
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	double x_diff, y_diff;

	int width, height;

	glfwGetWindowSize(window, &width, &height);

	x_diff = (xpos - last_xpos) / width;
	y_diff = (ypos - last_ypos) / height;

	if (x_diff != 0 || y_diff != 0) has_moved = true;

	// pitch
	rotation.x += y_diff * 30.0f;
	// yaw
	rotation.y += x_diff * 30.0f;

	// roll
	if (isPressed(GLFW_KEY_E)) {
		rotation.z += 1.0f;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_Q)) {
		rotation.z -= 1.0f;
		has_moved = true;
	}

	kernelInfo.camera_info.rotation = rotation;

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

	last_xpos = xpos;
	last_ypos = ypos;
	// **

	float SPEED_ = 0.125f * (t_diff / 20.0f);

	if (isPressed(GLFW_KEY_W)) {
		speed.z += SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_A)) {
		speed.x -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_S)) {
		speed.z -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_D)) {
		speed.x += SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_LEFT_CONTROL)) {
		speed.y -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_SPACE)) {
		speed.y += SPEED_;
		has_moved = true;
	}

	position += glm::cross(up, forward) * speed.x * 0.1f;
	position.y += speed.y * 0.1f;
	position += forward * -speed.z * 0.1f;

	for (int i = 0; i < 3; i++) {
		if (abs(speed[i]) < 0.05) {
			speed[i] = 0.0;
		}
		if (speed[i] > 0.0) {
			speed[i] -= 0.075 * (t_diff / 20.0f);
		}
		else if (speed[i] < 0.0) {
			speed[i] += 0.075 * (t_diff / 20.0f);
		}
	}

	kernelInfo.camera_info.origin = position;

	kernelInfo.set_camera(position, forward, up);
}