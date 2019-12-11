//  nvprof --print-gpu-summary ./main
#include <iostream>
#include "timer.h"   
#include "CONV_cuDNN.cuh"
#include "CONV_cuBLAS.cuh" 
#include "CONV_cuSPARSE.cuh" 
#include "CONV_CUDA.cuh" 
#include "CONV_proposed.cuh" 
#include "im2col.cuh" 

#include "CONV_ref.h"
#include <bits/stdc++.h> 

using namespace std;


//From Berkeley Vision's Caffe
//Refer to Caffe's license : https://github.com/BVLC/caffe/blob/master/LICENSE
static inline bool is_a_ge_zero_and_a_lt_b2(int a, int b) 
{
  return (unsigned int)a < (unsigned int)(b);
}

void Im2Col2(float *data_im, int channels, int height, int width, int kernel_h, int kernel_w,
            int pad_h, int pad_w, int stride_h, int stride_w, float *data_col) 
{
  int output_h = (height + 2 * pad_h - ((kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width + 2 * pad_w - ((kernel_w - 1) + 1)) / stride_w + 1;
  int channel_size = height * width;

  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) 
        {
          if (!is_a_ge_zero_and_a_lt_b2(input_row, height)) 
          {
            for (int output_cols = output_w; output_cols; output_cols--) 
            {
              *(data_col++) = 0;
            }
          } 
          else 
          {
            int input_col = -pad_w + kernel_col;
            for (int output_col = output_w; output_col; output_col--) 
            {
              if (is_a_ge_zero_and_a_lt_b2(input_col, width)) 
              {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
 
int main(int argc, char const *argv[])
{ 
  GpuTimer timer; 

  //struct TensorDim inputDims = {1,5,5,3}; // NHWC 
  struct TensorDim inputDims = {1,25,25,512}; // NHWC 
 
  int input_size= inputDims.n *inputDims.h * inputDims.w * inputDims.c;
  float* h_input =new float [input_size];
 
  init_matrix(h_input, inputDims); 
 
  //struct TensorDim KernelDims ={2,3,3,3}; // NHWC 
  struct TensorDim KernelDims = {512,3,3,512}; // NHWC 
  int kernel_size = KernelDims.n * KernelDims.h * KernelDims.w * KernelDims.c;
  float* h_kernels = new float [kernel_size];

  init_kernel_2(h_kernels, KernelDims, 0.75); 
  //std::copy(test_filter, test_filter+kernel_size, h_kernels); 
 
  struct TensorDim outputDims = {inputDims.n, inputDims.h, inputDims.w, KernelDims.n};  
  int output_size = outputDims.n *outputDims.h * outputDims.w * outputDims.c;

  //print_matrixNCHW(h_input, inputDims);
  //print_matrixNCHW(h_kernels, KernelDims);


  printf("---------------------------------------------\n");
  printf("Refernce Convolution output:\n");
  // Refernce Convolution output
  float* h_output_ref = new float [output_size];
  //memset(h_output_ref, 0, sizeof(h_output_ref));


  //RefConv2dF32(h_input, inputDims, h_kernels, KernelDims, h_output_ref, outputDims, 1.0, 1.0);
 

  bool pass; 
  //print_matrixNCHW(h_output_ref, outputDims); 

  printf("---------------------------------------------\n");
  printf("cuDNN Convolution output:\n");

  // cuDNN Convolution output
  float* h_output_cuDNN = new float [output_size];
  memset(h_output_cuDNN, 0, sizeof(h_output_cuDNN));  


 convolution_cudnn(h_input, inputDims, h_kernels, KernelDims, h_output_cuDNN); 
 //print_matrixNCHW(h_output_cuDNN, outputDims);
 //pass= TensorCompare(h_output_cuDNN, h_output_ref, outputDims);  
 printf("PASS [%d] | cuDNN Convolution.\n",pass);
 
  
  printf("---------------------------------------------\n");
  printf("cuBLAS Convolution output:\n");

  float *scratchpad = new float[(outputDims.h * outputDims.w * inputDims.c * KernelDims.w * KernelDims.h)];
  float* h_output_im2col = new float [output_size];

  RandInitF32(h_output_im2col, output_size);


  struct TensorDim scratchpadDims = {1, KernelDims.w * KernelDims.h * KernelDims.c, outputDims.h * outputDims.w, 1}; // NHWC 
 
  timer.Start(); 
  convolution_cuBLAS(h_input, h_kernels, scratchpad, inputDims, outputDims, scratchpadDims,
                      KernelDims, 1, 1,  h_output_im2col);
  timer.Stop(); 

  pass= TensorCompare(h_output_im2col, h_output_cuDNN, outputDims);  
  //print_matrixNCHW(h_output_im2col, outputDims);

  printf("PASS [%d] | cuBLAS Convolution: %f msecs.\n",pass, timer.Elapsed());
 

  printf("---------------------------------------------\n");
  printf("cuSPARSE Convolution output:\n");

  float* h_output_cuSPARSE = new float [output_size];
  
  timer.Start(); 
  convolution_cuSPARSE(h_input, h_kernels, inputDims, outputDims, KernelDims, h_output_cuSPARSE);
  timer.Stop();
  pass= TensorCompare(h_output_cuSPARSE, h_output_cuDNN, outputDims);  

  //print_matrixNCHW(h_output_cuSPARSE, outputDims);
  printf("PASS [%d] | cuSPARSE Convolution: %f msecs.\n",pass, timer.Elapsed());
  

  //struct TensorDim scratchpadDims = {1, KernelDims.w * KernelDims.h * KernelDims.c, outputDims.h * outputDims.w, 1}; // NHWC 
  //float *scratchpad = new float[(outputDims.h * outputDims.w * inputDims.c * KernelDims.w * KernelDims.h)];


  printf("---------------------------------------------\n");
  printf("CUDA Convolution output:\n");

  float* h_output_cuda = new float [output_size];
 
  convolution_cuda(h_input, h_kernels, scratchpad, inputDims, outputDims, scratchpadDims,
                      KernelDims, 1, 1,  h_output_cuda);
   pass= TensorCompare(h_output_cuda, h_output_cuDNN, outputDims);  
   //print_matrixNCHW2(scratchpad, scratchpadDims);

  //print_matrixNCHW(h_output_cuda, outputDims);
  printf("PASS [%d] | cuda Convolution:\t %f msecs.\n",pass);
 
 
  printf("---------------------------------------------\n");
  printf("structured Convolution output:\n");
   
 
  float* h_output_strcutured = new float [output_size];
  
  //for(int i=0;i <10; i++)
  convolution_proposed(h_input, h_kernels, h_output_strcutured, inputDims, outputDims, KernelDims, 1, 1);
 
   
  pass= TensorCompare(h_output_strcutured, h_output_cuDNN, outputDims);  

  //print_matrixNCHW(h_output_strcutured, outputDims);
  printf("PASS [%d] | cuda structured.\n",pass );
  
  /*
  delete[] h_input;
  delete[] h_kernels;
  delete[] h_output_strcutured;
  delete[] h_output_cuDNN;
  delete[] h_output_ref;
  delete[] scratchpad;
  delete[] h_output_cuSPARSE;
  */

  cout<<"Done!"<<endl;
  return 0;
}










 
 