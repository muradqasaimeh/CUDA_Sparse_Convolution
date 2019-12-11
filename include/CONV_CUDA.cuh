#include <string>
#include <stdio.h>
#include <cudnn.h>
#include "utils.h"
#include <iostream>
#include "timer.h"

void convolution_cuda(float *input, float *weight, float *scratchpad, TensorDim in_dim, TensorDim out_dim, TensorDim sp_dim,
                      TensorDim kernal_dim, int pad, int stride,  float *output);








