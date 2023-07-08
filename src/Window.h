#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>

#include "Shader.h"
#include "Quad.h"
#include "Input.h"

class Window {
public:
	unsigned int width;
	unsigned int height;

	Window(unsigned int width, unsigned int height);

	int init();
	void destroy();
private:
	GLFWwindow* _window;

	std::unique_ptr<Shader> _shader;
	std::unique_ptr<Shader> _accum_shader;
	std::unique_ptr<Quad> _blit_quad;
	std::unique_ptr<Quad> _accum_frame;
	std::unique_ptr<Quad> _current_frame;

	int init_glad();
	int init_glfw();
	int init_framebuffer();
	int init_quad();

	Input _input;
	int _frame_count;
	std::chrono::steady_clock::time_point _last_frame;


	void tick_input(float t_diff);
	void tick_render();
	void tick();

};