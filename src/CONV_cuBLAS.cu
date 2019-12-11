

#include "CONV_cuBLAS.cuh" 
using namespace std;

//From Berkeley Vision's Caffe
//Refer to Caffe's license : https://github.com/BVLC/caffe/blob/master/LICENSE
static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return (unsigned int)a < (unsigned int)(b);
}

void Im2Col(float *data_im, int channels, int height, int width, int kernel_h, int kernel_w,
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
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) 
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
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) 
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

  
void convolution_cuBLAS(float *input, float *weight, float *scratchpad, TensorDim in_dim, TensorDim out_dim, TensorDim sp_dim,
                      TensorDim kernal_dim, int pad, int stride,  float *output) 
{
  GpuTimer timer;

  float alpha = 1.0f;
  float beta  = 0.0f;

  timer.Start(); 
  Im2Col(input, in_dim.c, in_dim.h, in_dim.w, kernal_dim.h, kernal_dim.w, pad, pad, stride, stride, scratchpad); 
  timer.Stop();
  printf("Im2Col conversion:\t %f msecs.\n",timer.Elapsed());

 
 
  cublasHandle_t handle;
  cublasCreate(&handle);

  int A_wdith  = kernal_dim.c * kernal_dim.h * kernal_dim.w;
  int A_Height = kernal_dim.n;  

  int B_wdith  = sp_dim.w; 
  
  // allocate device memory
  float *d_A, *d_B, *d_C;  
  cudaMalloc(&d_A, TensorSize(kernal_dim)*sizeof(float));  
  cudaMalloc(&d_B, TensorSize(sp_dim)*sizeof(float));
  cudaMalloc(&d_C, TensorSize(out_dim)*sizeof(float));
  
  timer.Start(); 
  // copy host memory to device
  cudaMemcpy(d_A, weight, TensorSize(kernal_dim)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, scratchpad, TensorSize(sp_dim)*sizeof(float), cudaMemcpyHostToDevice);
  timer.Stop();

  printf("cudaMemcpyHostToDevice:\t %f msecs.\n", timer.Elapsed());


  //print_matrixNCHW(weight, kernal_dim);
  //print_matrixNCHW(scratchpad, sp_dim);

  //cublasOperation_t CUBLAS_OP_N; 
  timer.Start(); 
  cublasStatus_t ret = cublasSgemm( handle, 
                                    CUBLAS_OP_N, 
                                    CUBLAS_OP_N, 
                                    B_wdith,                /*Width  B*/ 
                                    A_Height,               /*Height A*/ 
                                    A_wdith,                /*Width  A*/ 
                                    &alpha,   
                                    d_B,                     /*d_B*/
                                    B_wdith,                 /*Width  B*/ 
                                    d_A,                     /*d_A*/
                                    A_wdith,                 /*Width  A*/
                                    &beta, 
                                    d_C,                      /*d_C*/
                                    B_wdith                   /*uiWB*/
                                    );

  timer.Stop();
  printf("cublas Convolution:\t %f msecs.\n",timer.Elapsed());
 // copy result from device to host


  timer.Start();   
  cudaMemcpy(output, d_C, TensorSize(out_dim)*sizeof(float), cudaMemcpyDeviceToHost); 
  timer.Stop();
  printf("cudaMemcpyDeviceToHost:\t %f msecs.\n", timer.Elapsed());

}
 

 






 






