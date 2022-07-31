#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "input.h"
#include "Quad.h"
#include "Shader.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// resize the frame buffer
	glViewport(0, 0, width, height);
}


Window::Window(unsigned int width, unsigned int height) {
	Window::width = width;
	Window::height = height;
}

int Window::init() {

	// initialize and configure glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// initialize and create window for glfw
	GLFWwindow* window = glfwCreateWindow(Window::width, Window::height, "LearnOpenGL", NULL, NULL);

	// handle error and exit
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// initialize glad before calling any opengl function
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// set dimensions of frame buffer and map coordinates
	glViewport(0, 0, Window::width, Window::height);

	// set callback function for when window gets resized
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	Shader* shader = new Shader("./shaders/rendertype_screen.vsh","./shaders/rendertype_screen.fsh");
	Quad* quad = new Quad(width, height);

	// main render loop
	while (!glfwWindowShouldClose(window))
	{
		// input
		processInput(window);


		// render
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shader->use();
		glBindVertexArray(quad->VAO);
		glBindTexture(GL_TEXTURE_2D, quad->texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);


		// check and call events and swap buffers between frames
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// terminate glfw and return success when exiting program
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;

}