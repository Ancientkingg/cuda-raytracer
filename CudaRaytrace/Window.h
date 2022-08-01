#pragma once

class Window {
public:
	unsigned int width;
	unsigned int height;
	Window(unsigned int width, unsigned int height);
	int init();
};