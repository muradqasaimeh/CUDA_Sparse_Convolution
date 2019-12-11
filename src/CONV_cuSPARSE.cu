

#include "CONV_cuSPARSE.cuh" 
#include "data_reshape.h"
using namespace std;


static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return (unsigned int)a < (unsigned int)(b);
}
 

void Im2Col_fun(float *data_im, int channels, int height, int width, int kernel_h, int kernel_w,
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

 

void transpose(float *src, float *dst, const int N, const int M) 
{
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}

void convolution_cuSPARSE(float *input, float *weight, TensorDim in_dim, TensorDim out_dim, TensorDim kernal_dim, float *output) 
{
    GpuTimer timer;

    //Initialize cuSPARSE
    cusparseHandle_t handle;    
    cusparseCreate(&handle);

    int Nrows= kernal_dim.n;
    int Ncols= kernal_dim.c * kernal_dim.h * kernal_dim.w;

    //print_matrixNCHW(weight, {1,Nrows,Ncols,1});
    float * weight_t= new float[TensorSize(kernal_dim)];
    transpose(weight, weight_t, Nrows, Ncols);

    //print_matrixNCHW(weight_t, {1,Ncols,Nrows,1});

    //create device array and copy host to it
    float *d_weight_dense;   
    cudaMalloc(&d_weight_dense, TensorSize(kernal_dim) * sizeof(float));
    cudaMemcpy(d_weight_dense, weight_t, TensorSize(kernal_dim) * sizeof(float), cudaMemcpyHostToDevice);


 
    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType      (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);  

    //-------------------------------------------------------------------------------------------------------------------------------
    int nnz = 0;                                // --- Number of nonzero elements in dense matrix
    int lda = Nrows;                            // --- Leading dimension of dense matrix 

    int *d_nnzPerVector;   
    cudaMalloc(&d_nnzPerVector, Nrows * sizeof(int));
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_weight_dense, Nrows, d_nnzPerVector, &nnz);
 

    // --- Host side number of nonzero elements per row
    int *h_nnzPerVector = (int *)malloc(Ncols * sizeof(*h_nnzPerVector));
    cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost);

    printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
    int sum=0;
    for (int i = 0; i < Nrows; ++i) 
      sum+=h_nnzPerVector[i];
    printf("Number of nonzero elements = %i \n",sum);
    printf("\n");


    // --- Device side dense matrix
    float *d_A;             cudaMalloc(&d_A, nnz * sizeof(float));
    int *d_A_RowIndices;    cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(int));
    int *d_A_ColIndices;    cudaMalloc(&d_A_ColIndices, nnz * sizeof(int));

    timer.Start();
    cusparseSdense2csr(handle, Nrows, Ncols, descrA, d_weight_dense, Nrows, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices);
    timer.Stop();
    printf("cusparseSdense2csr:\t %f msecs.\n",timer.Elapsed());

    // --- Host side dense matrix
    float *h_A = (float *)malloc(nnz * sizeof(float));     
    int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(int));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(int));

    //for (int i = 0; i < nnz; ++i) printf("A[%i] = %.04f ", i, h_A[i]); printf("\n");

    //for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

    //for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);   


    //-------------------------------------------------------------------------------------------------------------------------------

    float alpha =1.0;
    float beta  =0.0;
    float *d_B, *d_C;  


    float *scratchpad = new float[(out_dim.h * out_dim.w * in_dim.c * kernal_dim.w * kernal_dim.h)]; 
    struct TensorDim scratchpadDims = {1, kernal_dim.w * kernal_dim.h * kernal_dim.c, out_dim.h * out_dim.w, 1}; // NHWC 

    timer.Start();
    Im2Col_fun(input, in_dim.c, in_dim.h, in_dim.w, kernal_dim.h, kernal_dim.w, 1, 1, 1, 1, scratchpad); 
    timer.Stop();
    printf("Im2Col conversion:\t %f msecs.\n",timer.Elapsed());
    
    float * scratchpad_t= new float[TensorSize(scratchpadDims)];
    transpose(scratchpad, scratchpad_t, scratchpadDims.h, scratchpadDims.w);

    //print_matrixNCHW(scratchpad, scratchpadDims);
    //cout<<"--------"<<endl;
    //print_matrixNCHW(scratchpad_t, scratchpadDims);

    //-------------------------------------------------------------------------------------------------------------------------------

    cudaMalloc(&d_B, TensorSize(scratchpadDims)*sizeof(float));
    cudaMalloc(&d_C, TensorSize(out_dim)*sizeof(float));

    timer.Start();
    // copy host memory to device
    cudaMemcpy(h_A, d_A, nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_B, scratchpad_t, TensorSize(scratchpadDims)*sizeof(float), cudaMemcpyHostToDevice);
    timer.Stop();
    printf("cudaMemcpyHostToDevice:\t %f msecs.\n", timer.Elapsed());

    int n=scratchpadDims.w;
    int ldb= scratchpadDims.h;
    //printf("Nrows=%d , n=%d , ldb=%d  Ncols=%d , nnz=%d \n",Nrows, n,ldb, Ncols, nnz);

    timer.Start();
    cusparseStatus_t ret =cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                           Nrows, n, Ncols, nnz, &alpha, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_B, ldb, &beta, d_C, Nrows);
 
    timer.Stop();
    printf("cusparse Convolution:\t %f msecs.\n", timer.Elapsed());

  // return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
  //                        csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
 
/*
cusparseScsrmm(cusparseHandle_t         handle,
               cusparseOperation_t      transA,
               int                      m, //number of rows of sparse matrix A.
               int                      n, //number of columns of dense matrices B and C.
               int                      k, //number of columns of sparse matrix A.
               int                      nnz, //number of nonzero elements of sparse matrix A.
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             csrValA, //<type> array of nnz ( = csrRowPtrA(m) - csrRowPtrA(0) ) nonzero elements of matrix A.
               const int*               csrRowPtrA, //integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.
               const int*               csrColIndA, //integer array of nnz ( = csrRowPtrA(m) - csrRowPtrA(0) ) column indices of the nonzero elements of matrix A.
               const float*             B, //array of dimensions (ldb, n).
               int                      ldb, //leading dimension of B. It must be at least max (1, k) if op ( A ) = A and at least max (1, m) otherwise.
               const float*             beta, //array of dimensions (ldc, n).
               float*                   C,
               int                      ldc) //leading dimension of C. It must be at least max (1, m) if op ( A ) = A and at least max (1, k) otherwise.

 */
  float* h_output2 = new float[TensorSize(out_dim)];
 

  timer.Start();
  cudaMemcpy(h_output2, d_C, TensorSize(out_dim)*sizeof(float), cudaMemcpyDeviceToHost); 
  timer.Stop();
  printf("cudaMemcpyDeviceToHost:\t %f msecs.\n", timer.Elapsed());

  NHWC2NCHW(h_output2, out_dim.n, out_dim.c, out_dim.h, out_dim.w, output); 

}
  






