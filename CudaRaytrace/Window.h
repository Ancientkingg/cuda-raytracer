#pragma once

class Window {
	unsigned int width;
	unsigned int height;

public:
	Window(unsigned int width, unsigned int height);
	int init();
};