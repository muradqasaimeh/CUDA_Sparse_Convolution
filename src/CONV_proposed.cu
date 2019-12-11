
#include "CONV_proposed.cuh" 
#include "im2col.cuh" 
using namespace std;

#define TILE_DIM 2  // Tile dimension
#define TILE_DIM_2 4  // Tile dimension

int calcChannelDepth(float *weight, TensorDim kernal_dim);
void pre_process_weights(float *weight, TensorDim kernal_dim, float *weight_m, bool *map_bin);


float* im2colWithCuda(float* data_im, TensorDim in_dim, TensorDim kernal_dim, int pad, int stride, float* data_col)
{
  float *dev_image = 0;
  float *dev_col = 0; 
  
  int height_col = (in_dim.h + 2 * pad - kernal_dim.h) / stride + 1;
  int width_col  = (in_dim.w + 2 * pad - kernal_dim.h) / stride + 1;
    
  int K = kernal_dim.h*kernal_dim.w*in_dim.c; 
  int N = height_col*width_col;

  int image_size = in_dim.h * in_dim.w * in_dim.c;
  int images_size = image_size * in_dim.n; 
  int col_size = N*K; 

  // col 
  cudaMalloc((void**)&dev_col, N * K *in_dim.n * sizeof(float));
  
  // image
  cudaMalloc((void**)&dev_image, images_size* sizeof(float));
  cudaMemcpy(dev_image, data_im, images_size * sizeof(float), cudaMemcpyHostToDevice);
 
  float* t_dev_image = dev_image;
  float* t_dev_col = dev_col;

  for(int i = 0; i < in_dim.n; i++)
  {
    // Launch a kernel on the GPU with one thread for each element.
    im2col_gpu(t_dev_image, in_dim.c, in_dim.h, in_dim.w, kernal_dim.h, pad, stride, t_dev_col);

    //Perform warmup operation with cublas
    t_dev_image += image_size;
    t_dev_col += col_size; 
  }
 
  // Copy output vector from GPU buffer to host memory.
  cudaMemcpy(data_col, dev_col, N * K *in_dim.n* sizeof(float), cudaMemcpyDeviceToHost); 
 
  cudaFree(dev_image);
  return dev_col; 
} 


__global__ void MatMul2(float* weights, bool* map_bin, float* A, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols, int weights_k2C) 
{
  
  //printf("blockIdx.x= %d  blockIdx.y=%d  threadIdx.x=%d  threadIdx.y=%d  ACols=%d\n",blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, BCols);

  float CValue = 0;
    
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ float   As     [TILE_DIM][TILE_DIM];
  __shared__ float   Filter [TILE_DIM][TILE_DIM]; 
  __shared__ bool    Map    [TILE_DIM][TILE_DIM]; 
  __shared__ int     map_sum; 
  
 // map_sum=0;
 // __syncthreads();

  int current_f=0;
 
  for (int k = 0; k < ceilf((TILE_DIM + ACols - 1)/(float)TILE_DIM); k++) 
  {
    
    int index_y= k*TILE_DIM + threadIdx.y;

    if (index_y < BRows && Col < BCols)  
      As[threadIdx.y][threadIdx.x] = A[index_y*BCols + Col];
    else                          
      As[threadIdx.y][threadIdx.x] = 0.0;
 
    if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)  
      Map[threadIdx.y][threadIdx.x] = map_bin[Row*ACols + k*TILE_DIM + threadIdx.x];
    else                          
      Map[threadIdx.y][threadIdx.x] = 0.0;


    if(map_sum % (TILE_DIM_2)==0)
    {
      int index_f= current_f*TILE_DIM + threadIdx.x;

      if (index_f < weights_k2C && Row < ARows)  
        Filter[threadIdx.y][threadIdx.x] = weights[Row*weights_k2C + k*TILE_DIM + threadIdx.x];
      else                          
        Filter[threadIdx.y][threadIdx.x] = 0.0;  
        index_f++;
    }

    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n)
    { 
      if(Map[n][threadIdx.x]==1)
      {
          CValue += As[threadIdx.y][n] * Filter[n][threadIdx.x]; 
      } 
    }

  //  if((blockIdx.x==0)&&(blockIdx.y==0)&&(k==1))
  //    printf("k=%d blockIdx.x= %d  blockIdx.y=%d  threadIdx.x=%d  threadIdx.y=%d  CValue=%0.1f. \n",
   //         k, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,  CValue);


    if(Map[threadIdx.y][threadIdx.x]==1)
        atomicAdd(&map_sum,1);//map_sum= map_sum+1;

    __syncthreads();

  // if((blockIdx.x==0)&&(blockIdx.y==0))//&&(k<=2))
  //     printf("blockIdx.x= %d  blockIdx.y=%d  threadIdx.x=%d  threadIdx.y=%d k=%d As[%d][%d]=%0.1f Filter[%d][%d]=%0.1f. Map[%d][%d]=%d  map_sum=%d CValue=%0.1f \n",
  //      blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,k, threadIdx.y,threadIdx.x, As[threadIdx.y][threadIdx.x],
   //      threadIdx.y,threadIdx.x, Filter[threadIdx.y][threadIdx.x],threadIdx.y,threadIdx.x, Map[threadIdx.y][threadIdx.x], map_sum, CValue);
 

/*


    int index_y= k*TILE_DIM + threadIdx.y;

    if (index_y < BRows && Col < BCols)  
      Map[threadIdx.y][threadIdx.x] = map_bin[index_y*BCols + Col];
    else                          
      Map[threadIdx.y][threadIdx.x] = 0.0;  */

   

/*

    //printf("Thread x=%d y=%d As=%0.1f Bs_w=%0.1f Bs_c=%d\n", threadIdx.x,threadIdx.y, As[threadIdx.y][threadIdx.x], Bs_w[threadIdx.y][threadIdx.x],  Bs_c[threadIdx.y][threadIdx.x]);
*/
/*
    for (int n = 0; n < TILE_DIM; ++n)
    { 
    	if(n==Map[n][threadIdx.x])
    	{
      		CValue += As[threadIdx.y][n] * Filter[n][threadIdx.x];
      		//printf("Thread x=%d y=%d As=%0.1f Bs_w=%0.1f CValue=%0.1f\n", threadIdx.x,threadIdx.y, As[threadIdx.y][n], Bs_w[n][threadIdx.x], CValue);
      		map_sum= map_sum+1;

    	}
    }*/

   // __syncthreads();
  } 
  
  if (Row < CRows && Col < CCols) 
    C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue; 
}


void convolution_proposed(float *input, float *weight, float *output, TensorDim in_dim, TensorDim out_dim, TensorDim kernal_dim, int pad, int stride)
{ 

  //print_matrixNCHW(weight, kernal_dim);

  float *scratchpad = new float[(out_dim.h * out_dim.w * in_dim.c * kernal_dim.w * kernal_dim.h)]; 
  struct TensorDim scratchpadDims = {1, kernal_dim.w * kernal_dim.h * kernal_dim.c, out_dim.h * out_dim.w, 1}; // NHWC 

  float *dev_col = 0; 
  dev_col= im2colWithCuda(input, in_dim, kernal_dim, pad, stride, scratchpad); 
 
 
  // define weights and channel tensor
  int channel_depth= calcChannelDepth(weight, kernal_dim);
  float *weight_m=new float [channel_depth*kernal_dim.n*kernal_dim.w*kernal_dim.h];
  bool *map_bin =new bool [TensorSize(kernal_dim)];

  pre_process_weights(weight, kernal_dim, weight_m, map_bin);
	
  int weights_k2C= channel_depth*kernal_dim.w*kernal_dim.h;
  /*
  for(int i=0; i< channel_depth*kernal_dim.n*kernal_dim.w*kernal_dim.h; i++)
  { 
  	printf("weight_m[%d]= %0.1f\n",i, weight_m[i]);
  }
*/
  for(int i=0; i< kernal_dim.n; i++)
  { 
   //   for(int j=0; j< kernal_dim.c*kernal_dim.h*kernal_dim.w; j++)
  //      printf("%d ", map_bin[i*kernal_dim.c*kernal_dim.h*kernal_dim.w+j]);
  //    printf("\n");
  }

 
  //print_matrixNCHW_b(map_bin, kernal_dim);
 
  //print_matrixNCHW2(scratchpad, scratchpadDims);
 
  // allocate device memory
  float  *d_weights, *d_output;  
  bool   *d_map_bin;

  cudaMalloc(&d_weights, channel_depth*kernal_dim.n*kernal_dim.w*kernal_dim.h*sizeof(float));  
  cudaMalloc(&d_map_bin, TensorSize(kernal_dim)*sizeof(bool));   
  cudaMalloc(&d_output, TensorSize(out_dim)*sizeof(float));

  cudaMemcpy(d_weights, weight_m, channel_depth*kernal_dim.n*kernal_dim.w*kernal_dim.h*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_map_bin, map_bin, TensorSize(kernal_dim)*sizeof(bool), cudaMemcpyHostToDevice);
 
  int A_wdith  = kernal_dim.c * kernal_dim.h * kernal_dim.w;
  int A_Height = kernal_dim.n;  

  int B_wdith  = scratchpadDims.w; 

  // Invoke kernel 
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid; 
  dimGrid.x = (B_wdith + dimBlock.x - 1)/dimBlock.x;
  dimGrid.y = (A_Height + dimBlock.y - 1)/dimBlock.y;

  printf("A_Height =%d A_wdith=%d B_wdith=%d \n", A_Height, A_wdith, B_wdith);
  printf("dimGrid.x =%d dimGrid.y=%d dimBlock.x=%d. dimBlock.y=%d \n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

  MatMul2<<<dimGrid, dimBlock>>>(d_weights, d_map_bin, dev_col, d_output, A_Height, A_wdith, A_wdith, B_wdith, A_Height, B_wdith, weights_k2C); 
 
  printf("outputSIZE=%d\n", TensorSize(out_dim));

  // copy result from device to host
  cudaMemcpy(output, d_output, TensorSize(out_dim)*sizeof(float), cudaMemcpyDeviceToHost); 

  //print_matrixNCHW (output, out_dim);

 

}

int calcChannelDepth(float *weight, TensorDim kernal_dim)
{
	// compute # of non-zero weights
 	int cnt=0;
  	for(int k=0; k< kernal_dim.n; k++)
 	{
 		for(int i=0; i< kernal_dim.w; i++)
	  	{
	  		for (int j=0; j< kernal_dim.h; j++)
	  		{
	  			for (int c=0; c< kernal_dim.c; c++)
	  			{
	  				if(weight[(k*kernal_dim.c*kernal_dim.w*kernal_dim.h +c*kernal_dim.w*kernal_dim.h + j * kernal_dim.w + i)]!=0)
	  				{
	  					cnt++;
	  				}
	  			}
	  		}
	  	}
	}//end 
 
	return ceilf((cnt*1.0) /(kernal_dim.n *kernal_dim.w * kernal_dim.h));
}
void pre_process_weights(float *weight, TensorDim kernal_dim, float *weight_m, bool *map_bin)
{ 
	int cnt1=0;
	int cnt2=0;
  	for(int k=0; k< kernal_dim.n; k++)
 	{
 		for (int c=0; c< kernal_dim.c; c++)
	  	{
	 		for(int j=0; j< kernal_dim.h; j++)
		  	{
		  		for (int i=0; i< kernal_dim.w; i++)
		  		{
		  			if(weight[(k*kernal_dim.c*kernal_dim.w*kernal_dim.h +c*kernal_dim.w*kernal_dim.h + j * kernal_dim.w + i)]!=0)
		  			{
		  				weight_m[cnt1]=weight[(k*kernal_dim.c*kernal_dim.w*kernal_dim.h +c*kernal_dim.w*kernal_dim.h + j * kernal_dim.w + i)];
		  				map_bin[cnt2]=  1;  
		  				cnt1++;
		  				cnt2++;
		  			}
		  			else
		  			{
		  				map_bin[cnt2]=  0;  
		  				cnt2++;

		  			}
		  		} 
		  	}
	  	}
	}
} 




