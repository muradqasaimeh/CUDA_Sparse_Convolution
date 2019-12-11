#ifndef UTILS_H__
#define UTILS_H__
#include "timer.h"  
#include <string>
#include <stdio.h>
#include <cudnn.h>
#include <iostream>
//#include <cblas.h>
 

struct TensorDim{
  int n;
  int h;
  int w;
  int c;
};

#define checkCUDNN(expression)                               \
 {                                                           \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

// Initializes a 4D Tensor
void init_matrix(float *m, TensorDim mDims);
void init_kernel(float *m, TensorDim mDims, float sparsity);
void init_kernel_2(float *m, TensorDim mDims, float sparsity);

// print a 4D Tensor in NHWC
void print_matrix(float *m, TensorDim mDims);


void print_matrixNCHW(float *m, TensorDim mDims);
void print_matrixNCHW_b(bool *m, TensorDim mDims);

void print_matrixNCHW2(float *m, TensorDim mDims);

bool TensorCompare(float *m1, float *m2, TensorDim dim);

int TensorSize(TensorDim dim);

void RandInitF32(float *p_data, int N);


/*
// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);



// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
                const int height, const int width, const int ksize, const int pad,
                const int stride, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
                 const int height, const int width, const int ksize, const int pad,
                 const int stride, double* data_col);
*/
#endif  
 











