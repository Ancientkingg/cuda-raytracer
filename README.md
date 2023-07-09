# A real-time GPU accelerated ray tracer written in C++ and CUDA
***A real-time GPU accelerated ray tracer written in C++ and CUDA with the help of OpenGL for rendering to the screen.***


Before doing this CUDA project, I had a little bit of experience in the graphics world.
Namely, I had written [a '*real-time*' ray tracer](https://github.com/Ancientkingg/rust-raytracer) in Rust which ran on the CPU. 
In that project I managed to finish the [first book](https://raytracing.github.io/books/RayTracingInOneWeekend.html) of the 'Raytracing in X' series, which was a great learning experience.
After finishing however, I wanted to expand upon my capabilities and create a more complex ray tracer which maybe was also a bit more '*real-time*' than the one written in Rust.
I found out about CUDA and decided to use it to write a GPU-accelerated ray tracer with more features. Staring at a picture of a sphere on a plane seems a bit boring to me though, 
so I decided to make the renderer real-time with the help of OpenGL.


The project uses CMake as the build system to support portability and for me personally to get more familiar with CMake.
My aim for this project was for others to be able to simply clone the repository and build the project without much fuss.


## Development walkthrough

### Getting started
Before getting started with actually writing the renderer, I installed the latest version of the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) 
and created a CMake project in Visual Studio 2022.

The next step in the process was then to configure CMake to include the necessary libraries for OpenGL to work correctly.
I found a [very nice template repository](https://github.com/Shot511/OpenGLSampleCmake) that had everything I was looking for,
with which I configured my own CMake project.

After setting up the necessary ground work, I started with getting a simple window to draw to the screen.
With the help of the [OpenGL tutorial](https://learnopengl.com/Getting-started/Creating-a-window) from [LearnOpenGL.com](https://learnopengl.com/), I managed to get a simple window up and running.

![Window with black screen](/docs/images/black_window.png)

At the moment, the window isn't showing anything all too exciting though, so let's change that by adding a quad to the screen onto which we can draw an image.

### Drawing a quad
I set up a `Quad` class along with a `Shader` class that I got from the [OpenGL template repository](https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/shader.h).
The `Quad` class sets up a quad with the help of a vertex array object (VAO) and a vertex buffer object (VBO) and draws it to the screen.
The `Shader` class compiles a vertex and fragment shader and links them together into a shader program.

Normally the quad would draw the ray traced output image as a texture, but since no ray tracing has been implemented yet at this point. 
I decided to modify the fragment shader to draw a simple gradient to showcase my progress.
```glsl
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

void main()
{
    vec3 col = texture(screenTexture, TexCoords).rgb;
    FragColor = vec4(TexCoords.x, TexCoords.y, 0.0, 1.0);
}
```

This results in the following image being drawn to the screen:

![Window with gradient](/docs/images/gradient_window.png)

The plan is to let CUDA draw the ray traced image to a texture. 
We then use OpenGL to draw that texture to the quad on the screen.
This also enables us to add post-processing on top of the image in later stages.

### The ray tracing kernel
The next step is to write the ray tracing kernel in CUDA.
The kernel will basically do all of the ray tracing, output the result to a buffer and hand it over to OpenGL to draw to the screen.

I found a nice guide that adapted the [first book](https://raytracing.github.io/books/RayTracingInOneWeekend.html) of the 'Raytracing in X' series to CUDA.
I used [this guide](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) as a starting point for my own kernel, which I then adapted to my own needs.

After working through both guides and some debugging of my own I managed to get the window to draw a scene in real-time.

![Window showing a ray traced scene of spheres](/docs/images/raytraced_window.png)

### A movable camera
Even though our renderer is real-time now, it is still not very interactive.
The next step was to make the camera rotatable and movable, so that we can actually look around and enjoy the scene from different angles.
The camera supports WASD to move horizontally, Q and E to roll the camera, and, CTRL and SPACE to move vertically.
It uses a sort of motion based system to move the camera, which makes the movement feel somewhat natural.

https://github.com/Ancientkingg/CudaRaytrace/assets/67058024/333f98e2-c043-4ecf-881c-7044380091dd

The video above is a recording of the window with a resolution of 800x600 with 3 samples per pixel.
It runs at about 80-100 fps on my machine with an RTX 3070. Even though each pixel samples 3 rays, there is still some
visible noise in the video, *even with all of the video compression*, especially close around the spheres.
To somewhat mitigate this, my plan was to add a very crude implementation of temporal accumulation.

### Stacking frames
I am not an expert in graphics programming so there will most likely be some mistakes in my explanation, but [temporal accumulation](https://teamwisp.github.io/research/temporal_accumulation.html) 
in essence is a way to reduce noise in a rendered image by averaging out multiple frames. You can compare it to a long exposure shot in photography.
Each frame is rendered and stacked on top of each other, which results in a smoother image with less visible noise. 

One problem that arises from this, is that with a moving camera, frames will not align (perfectly) leaving us with a blurry mess.
There are some advanced techniques to reproject frames between positions, but since I am pretty novice in this area, I decided to go for the simplest approach that I could think of.

The renderer will accumulate frames as long as the camera is not moving. 
Once it moves (or rotates), the accumulated frames will be discarded and the renderer will (try to) start accumulating frames again.
To implement temporal accumulation I decided to make use of the shader pipeline we set up earlier.

The ray tracing kernel will continuously output frames to a buffer, which will then be used as a texture by OpenGL to draw to the screen.
A second buffer is used to accumulate frames. This buffer is also used as a texture, but it is not drawn directly to the screen.

https://github.com/Ancientkingg/CudaRaytrace/assets/67058024/0c14d1b8-510c-4a97-b8dc-bf65aa019d9d

As you can see after a set amount of time the image seems to clear up and the noise starts to fade away.
The accumulated frames are discarded as soon as the camera moves or rotates, so that we don't get a blurry mess like this:

https://github.com/Ancientkingg/CudaRaytrace/assets/67058024/79fe5447-ee7f-42bf-9b3d-b24015b5a090

### Improving code quality
At this point in the project, I realized that the code quality was not very good. 
When I started writing this ray tracer I was still very novice in writing actual C++ code, which resulted in a lot of bad practices creeping into the code.
Particularly, the high usage of raw device pointers and the lack of proper memory management was a big problem.
Before continuing with the project, I decided to try to rewrite the code to improve memory management and to make the code more readable.

I used valgrind and Nvidia's compute sanitizer to detect memory leaks and managed to fix all of them by using RAII.

