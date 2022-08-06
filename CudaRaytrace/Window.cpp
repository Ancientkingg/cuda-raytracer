#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <chrono>

#include "Window.h"
#include "input.h"
#include "Quad.h"
#include "Shader.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// resize the frame buffer
	glViewport(0, 0, width, height);
	Window* myWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

	myWindow->width = width;
	myWindow->height = height;
}


Window::Window(unsigned int width, unsigned int height) {
	Window::width = width;
	Window::height = height;
}

void copyFrameBufferTexture(int width, int height, int fboIn, int textureIn, int fboOut, int textureOut)
{
	// Bind input FBO + texture to a color attachment
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fboIn);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIn, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	// Bind destination FBO + texture to another color attachment
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboOut);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureOut, 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT1);

	// specify source, destination drawing (sub)rectangles.
	glBlitFramebuffer(0, 0, width, height,
		0, 0, width, height,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	// unbind the color attachments
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
}

int Window::init() {

	// initialize and configure glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// initialize and create window for glfw
	GLFWwindow* window = glfwCreateWindow(Window::width, Window::height, "A CUDA raytracer", NULL, NULL);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetWindowUserPointer(window, reinterpret_cast<void*>(this));

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

	Quad* blit = new Quad(Window::width, Window::height);
	blit->makeFBO();

	Shader* shader = new Shader("./shaders/rendertype_screen.vsh","./shaders/rendertype_screen.fsh");
	Quad* currentFrame = new Quad(Window::width, Window::height);
	currentFrame->cudaInit(Window::width, Window::height);
	currentFrame->makeFBO();

	Shader* accumulateShader = new Shader("./shaders/rendertype_accumulate.vsh", "./shaders/rendertype_accumulate.fsh");
	Quad* accumulateFrame = new Quad(Window::width, Window::height);
	accumulateFrame->makeFBO();

	accumulateShader->use();
	glUniform1i(glGetUniformLocation(accumulateShader->ID, "currentFrameTex"), 0);
	glUniform1i(glGetUniformLocation(accumulateShader->ID, "lastFrameTex"), 1);
	int frameCount = 1;

	std::chrono::steady_clock::time_point last_frame = std::chrono::steady_clock::now();

	Input input;

	

	// main render loop
	while (!glfwWindowShouldClose(window))
	{
		std::chrono::steady_clock::time_point this_frame = std::chrono::steady_clock::now();
		float t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(this_frame - last_frame).count();
		last_frame = this_frame;
		
		//input
		input.processQuit(window);
		input.processCameraMovement(window, currentFrame->renderer, t_diff);
		if (input.hasCameraMoved()) frameCount = 1;

		// render
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// render current frame
		glBindFramebuffer(GL_FRAMEBUFFER, currentFrame->framebuffer);
		currentFrame->renderKernel(Window::width, Window::height);
		shader->use();
		glBindVertexArray(currentFrame->VAO);
		glBindTexture(GL_TEXTURE_2D, currentFrame->texture);
		glDrawArrays(GL_TRIANGLES, 0, 6); // draw current frame to texture

		// copy accumulated frames to another texture so that we can sample it
		copyFrameBufferTexture(Window::width, Window::height, accumulateFrame->framebuffer, accumulateFrame->texture, blit->framebuffer, blit->texture);

		// composite the accumulated frames with the current one
		glBindFramebuffer(GL_FRAMEBUFFER, accumulateFrame->framebuffer);
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, currentFrame->texture);
		glActiveTexture(GL_TEXTURE0 + 1);	
		glBindTexture(GL_TEXTURE_2D, blit->texture);
		accumulateShader->use();
		glUniform1i(glGetUniformLocation(accumulateShader->ID, "frameCount"), frameCount);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		// render result to screen
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		shader->use();
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, accumulateFrame->texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		

		// check and call events and swap buffers between frames
		glfwSwapBuffers(window);
		glfwPollEvents();

		frameCount++;
	}

	// terminate glfw and return success when exiting program
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;

}