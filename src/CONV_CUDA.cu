

#include "CONV_CUDA.cuh" 
using namespace std;
#define TILE_DIM 2           // Tile dimension
 

//From Berkeley Vision's Caffe
//Refer to Caffe's license : https://github.com/BVLC/caffe/blob/master/LICENSE
static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return (unsigned int)a < (unsigned int)(b);
}

void Im2Col_Function(float *data_im, int channels, int height, int width, int kernel_h, int kernel_w,
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



__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) 
{
  
  float CValue = 0;
    
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];
     
  for (int k = 0; k < ceilf((TILE_DIM + ACols - 1)/(float)TILE_DIM); k++) 
  {
      
    if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)  
      As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
    else                          
      As[threadIdx.y][threadIdx.x] = 0.0;

    //if((blockIdx.x==0)&&(blockIdx.y==0)&&(k==0))
    //    printf("blockIdx.x= %d  blockIdx.y=%d  threadIdx.x=%d  threadIdx.y=%d (Row*ACols + index_x)=%d As[%d][%d]=%0.1f\n",blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, (Row*ACols + k*TILE_DIM + threadIdx.x), threadIdx.y,threadIdx.x,As[threadIdx.y][threadIdx.x]);


    if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)  
      Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
    else                          
      Bs[threadIdx.y][threadIdx.x] = 0.0;
         
    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n) 
      CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
    
    __syncthreads();
  }
    
  if (Row < CRows && Col < CCols) 
    C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

 
void convolution_cuda(float *input, float *weight, float *scratchpad, TensorDim in_dim, TensorDim out_dim, TensorDim sp_dim,
                      TensorDim kernal_dim, int pad, int stride,  float *output) 
{
  GpuTimer timer;
 
  timer.Start(); 
  Im2Col_Function(input, in_dim.c, in_dim.h, in_dim.w, kernal_dim.h, kernal_dim.w, pad, pad, stride, stride, scratchpad); 
  timer.Stop();
  printf("Im2Col conversion:\t %f msecs.\n",timer.Elapsed());


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
 
  // Invoke kernel 
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid; 
  dimGrid.x = (B_wdith + dimBlock.x - 1)/dimBlock.x;
  dimGrid.y = (A_Height + dimBlock.y - 1)/dimBlock.y;

  //print_matrixNCHW(weight, kernal_dim);
  //print_matrixNCHW(scratchpad, sp_dim);
 
  timer.Start(); 
  MatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, A_Height, A_wdith, A_wdith, B_wdith, A_Height, B_wdith); 

  timer.Stop();
  printf("CUDA convolution:\t %f msecs.\n",timer.Elapsed());
  // copy result from device to host

  timer.Start();   
  cudaMemcpy(output, d_C, TensorSize(out_dim)*sizeof(float), cudaMemcpyDeviceToHost); 
  timer.Stop();
  printf("cudaMemcpyDeviceToHost:\t %f msecs.\n",timer.Elapsed());

}
 
 


   





