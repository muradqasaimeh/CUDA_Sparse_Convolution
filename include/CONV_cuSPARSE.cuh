#include <stdio.h> 
#include "utils.h"
#include <iostream>
#include "timer.h"
#include <cusparse_v2.h>

void convolution_cuSPARSE(float *input, float *weight, TensorDim in_dim, TensorDim out_dim, TensorDim kernal_dim, float *output);