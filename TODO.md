# TODO
- [x] Convert raw device pointer usage to use `thrust::device_ptr`
- [x] Make use of `thrust::device_vector` instead of raw arrays for the world list
    - This does not seem to be possible since `thrust::device_vector` is designed to manage device memory **from the host**
- [x] Fix loss of data warnings in Camera class
- [x] Fix exception thrown when resizing window
- [ ] Fix CUDA memory leak related to `c`
- [ ] Check out how the OpenGL and CUDA interop works since right now a new frame buffer is allocated each frame (seems pretty bad for performance)
- [ ] Add a GUI to display information (imgui)
