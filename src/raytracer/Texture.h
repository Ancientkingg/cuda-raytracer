
#include "device_launch_parameters.h"
#include <glm/vec3.hpp>


class Texture {
public:
	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const = 0;
};

class SolidColor : public Texture {
public:
	__device__ SolidColor() {}
	__device__ SolidColor(glm::vec3 c) : color_value(c) {}
	__device__ SolidColor(float red, float green, float blue) : SolidColor(glm::vec3(red, green, blue)) {}


	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const override {
		return color_value;
	}

private:
	glm::vec3 color_value;
};

class CheckerTexture : public Texture {
public:
	__device__ CheckerTexture() {}
	__device__ CheckerTexture(Texture* even_tex, Texture* odd_tex) : even(even_tex), odd(odd_tex) {}
	__device__ CheckerTexture(glm::vec3 even_color, glm::vec3 odd_color) : even(new SolidColor(even_color)), odd(new SolidColor(odd_color)) {}

	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const override {
		double sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

	Texture* even;
	Texture* odd;

	__device__ ~CheckerTexture() {
		delete even;
		delete odd;
	}
};