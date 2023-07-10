#pragma once

#include "cuda_runtime.h"

#define check_cuda_errors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);