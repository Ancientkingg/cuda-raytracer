# TODO
- [ ] Convert raw device pointer usage to use `thrust::device_ptr`
- [ ] Make use of `thrust::device_vector` instead of raw arrays for the world list
    - This does not seem to be possible since `thrust::device_vector` is designed to manage device memory **from the host**
- [ ] Fix loss of data warnings in Camera class
- [ ] Start making use of smart pointers instead of using `cudaFree` and `cudaMalloc` directly to update code to a more modern C++ style
- [ ] Start making use of smart pointers inside of the ray tracing code** to update the code to a more modern C++ style
- [ ] Check out how the OpenGL and CUDA interop works since right now a new frame buffer is allocated each frame (seems pretty bad for performance)
- [ ] Add a GUI to display information (imgui)
