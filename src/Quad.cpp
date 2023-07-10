#include "Quad.h" 

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include "raytracer/kernel.h"
#include "cuda_errors.h"

#include <memory>

Quad::Quad(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    vertices = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);

    // bind Vertex Array Object
    glBindVertexArray(VAO);

    // bind reference to buffer on gpu
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // copy vertex data to buffer on gpu
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // set our vertex attributes pointers
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Quad::cuda_init() {
    check_cuda_errors(
        cudaGraphicsGLRegisterBuffer(&CGR,
            PBO,
            cudaGraphicsRegisterFlagsNone));
    _renderer = std::make_unique<KernelInfo>(this->CGR, width, height);
}

void Quad::cuda_destroy() {
    check_cuda_errors(
        cudaGraphicsUnregisterResource(CGR));
}

void Quad::make_FBO() {
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Quad::render_kernel() {
    glBindTexture(GL_TEXTURE_2D, 0);
    _renderer->render();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->PBO);
    glBindTexture(GL_TEXTURE_2D, this->texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void Quad::resize(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);    

    if (_renderer != nullptr) {
        check_cuda_errors(
            cudaGraphicsUnregisterResource(CGR));

        check_cuda_errors(
            cudaGraphicsGLRegisterBuffer(&CGR,
                PBO,
                cudaGraphicsRegisterFlagsNone));

        _renderer->resources = this->CGR;
        _renderer->resize(width, height);
    }
}