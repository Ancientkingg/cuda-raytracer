#pragma once

#include "device_launch_parameters.h"
#include <glm/vec3.hpp>

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const glm::vec3& origin, const glm::vec3& direction) { this->origin = origin; this->direction = direction; }
    __device__ glm::vec3 at(float t) const { return origin + t * direction; }

    glm::vec3 origin;
    glm::vec3 direction;
};