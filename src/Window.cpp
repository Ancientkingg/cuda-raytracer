#include "Window.h"

#include <iostream>
#include <iomanip>
#include <memory>

Window::Window(unsigned int width, unsigned int height) {
	Window::width = width;
	Window::height = height;
	Window::_frame_count = 0;
}

int Window::init_glfw() {
	// Initialize and configure GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Initialize and create window for GLFW
	_window = glfwCreateWindow(Window::width, Window::height, "A CUDA ray tracer", NULL, NULL);

	// Hides the cursor and captures it
	glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetWindowUserPointer(_window, reinterpret_cast<void*>(this));

	// Check if window was created
	if (_window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(_window);

	return 0;
}

int Window::init_glad() {
	// Initialize GLAD before calling any OpenGL function
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	return 0;
}

void Window::resize(unsigned int w, unsigned int h) {
	this->width = w;
	this->height = h;
	this->_current_frame->resize(w, h);
	this->_accum_frame->resize(w, h);
	this->_blit_quad->resize(w, h);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// resize the frame buffer
	glViewport(0, 0, width, height);
	Window* myWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

	myWindow->resize(width, height);
}

int Window::init_framebuffer() {
	// Set dimensions of frame buffer and map coordinates
	glViewport(0, 0, Window::width, Window::height);

	// Set callback function for when window gets resized
	glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);

	return 0;
}

int Window::init_quad() {
	
	_blit_quad = std::make_unique<Quad>(Window::width, Window::height);
	_blit_quad->make_FBO();

	_shader = std::make_unique<Shader>("./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_current_frame = std::make_unique<Quad>(Window::width, Window::height);
	_current_frame->cuda_init();
	_current_frame->make_FBO();

	_accum_shader = std::make_unique<Shader>("./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");
	_accum_frame = std::make_unique<Quad>(Window::width, Window::height);
	_accum_frame->make_FBO();

	_accum_shader->use();
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "currentFrameTex"), 0);
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "lastFrameTex"), 1);

	return 0;
}

int Window::init() {

	if (init_glfw() != 0) return -1;
	if (init_glad() != 0) return -1;
	if (init_framebuffer() != 0) return -1;
	if (init_quad() != 0) return -1;

	_last_frame = std::chrono::steady_clock::now();

	while (!glfwWindowShouldClose(_window)) {
		Window::tick();
	}

	Window::destroy();
	return 0;
}

void Window::destroy() {

	// Terminate CUDA allocated buffer
	_current_frame->cuda_destroy();

	// Terminate GLFW
	glfwDestroyWindow(_window);
	glfwTerminate();
}

void Window::tick_input(float t_diff) {

	//input
	_input.process_quit(_window);
	_input.process_camera_movement(_window, *(_current_frame->_renderer), t_diff);
	if (_input.has_camera_moved()) _frame_count = 1;
}

void copyFrameBufferTexture(int width, int height, int fboIn, int textureIn, int fboOut, int textureOut) {
	// Bind input FBO + texture to a color attachment
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fboIn);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIn, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	// Bind destination FBO + texture to another color attachment
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboOut);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureOut, 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT1);

	// Specify source, destination drawing (sub)rectangles.
	glBlitFramebuffer(0, 0, width, height,
		0, 0, width, height,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	// Unbind the color attachments
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
}

void Window::tick_render() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Render current frame
	glBindFramebuffer(GL_FRAMEBUFFER, _current_frame->framebuffer);
	_current_frame->render_kernel();
	_shader->use();
	glBindVertexArray(_current_frame->VAO);
	glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
	glDrawArrays(GL_TRIANGLES, 0, 6); // draw current frame to texture

	// Copy accumulated frames to another texture so that we can sample it
	copyFrameBufferTexture(Window::width, Window::height, _accum_frame->framebuffer, _accum_frame->texture, _blit_quad->framebuffer, _blit_quad->texture);

	// Composite the accumulated frames with the current one
	glBindFramebuffer(GL_FRAMEBUFFER, _accum_frame->framebuffer);
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, _blit_quad->texture);
	_accum_shader->use();
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "frameCount"), _frame_count);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Render result to screen
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glDrawArrays(GL_TRIANGLES, 0, 6);


	// Check and call events and swap buffers between frames
	glfwSwapBuffers(_window);
	glfwPollEvents();
}

void Window::tick() {

	std::chrono::steady_clock::time_point this_frame = std::chrono::steady_clock::now();
	float t_diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_frame - _last_frame).count();
	_last_frame = this_frame;

	// Print FPS
	//std::cout << "\r" << std::fixed << std::setprecision(2) << 1000.0 / t_diff << " fps";

	// Input
	tick_input(t_diff);

	 // Render
	tick_render();

	// Accumulate the amount of frames rendered
	_frame_count++;
}