
 
#include <cublas_v2.h> 
#include "utils.h"   


void convolution_cuBLAS(float *input, float *weight, float *scratchpad, TensorDim in_dim, TensorDim out_dim, TensorDim sp_dim,
                      TensorDim kernal_dim, int pad, int stride,  float *output);