#include <string>
#include <stdio.h>
#include <cudnn.h>
#include "utils.h"
#include <iostream>
#include "timer.h"


void convolution_proposed(float *input, float *weight, float *output, TensorDim in_dim, TensorDim out_dim, TensorDim kernal_dim, int pad, int stride);