#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>      
#include <cudnn.h>   
#include <vector>
#include <iostream>
#include "utils.h"
//#include <cblas.h>

using namespace std;
    

// Initializes a 4D Tensor
void init_matrix(float *m, TensorDim mDims)
{
 	for (int n=0; n< mDims.n; n++){
    	for (int i = 0; i <mDims.h; i++){
      		for (int j = 0; j <mDims.w; j++){
        		for (int k = 0; k <mDims.c; k++){ 
        			int index =   ((n*mDims.h+i) *mDims.w +j) *mDims.c + k;
        			m[index] = rand() % 10;   
      			}
      		}
      	}
	}
}

// Initializes a 4D Tensor
void init_kernel(float *m, TensorDim mDims, float sparsity)
{
  for (int n=0; n< mDims.n; n++)
  {
      for (int i = 0; i <mDims.h; i++){
          for (int j = 0; j <mDims.w; j++){
            for (int k = 0; k <mDims.c; k++){ 
              int index =   ((n*mDims.h+i) *mDims.w +j) *mDims.c + k;
              m[index] = (rand() % 10)+1;   
            }
          }
        }
  }
  int NZ= TensorSize(mDims)* sparsity; // number of zero values
  
  vector<int> v(TensorSize(mDims));

  //make vector
  for (size_t i = 0; i < TensorSize(mDims); ++i)
    v[i] = static_cast<int>(i);

  //shuffle
  for (size_t i = TensorSize(mDims) - 1; i > 0; --i)
    swap(v[i], v[static_cast<size_t>(rand()) % (i + 1)]);

  //print
  for (size_t i = 0; i < NZ; ++i) 
  { 
     m[v[i]]=0;;
  }
 
}


// Initializes a 4D Tensor
void init_kernel_2(float *m, TensorDim mDims, float sparsity)
{
  int NZ= ceilf(1.0*mDims.c* (1-sparsity)); 
  //printf("init_kernel2 NZ= %d\n",NZ );
  for (int n=0; n< mDims.n; n++)
  {
      for (int i = 0; i <mDims.h; i++)
      {
          for (int j = 0; j <mDims.w; j++)
          {
              vector<int> v(mDims.c);
              //make vector
              for (size_t i2 = 0; i2 < mDims.c; ++i2)
                v[i2] = static_cast<int>(i2);

              //shuffle
              for (size_t i3 =mDims.c - 1; i3 > 0; --i3)
                swap(v[i3], v[static_cast<size_t>(rand()) % (i3 + 1)]);

              for (int k = 0; k <mDims.c; k++)
              {

                //int index = ((n*mDims.h+i) *mDims.w +j) *mDims.c + k;
                int index =n* mDims.w*mDims.h*mDims.c + k*mDims.w*mDims.h+ i*mDims.w  + j;

                if(v[k]<NZ)
                {
                  m[index] = (rand() % 10)+1;   
                }
                else
                {
                  m[index] = 0;   

                }
              } 
          }
      }
  } 
}


// print a 4D Tensor in NCHW
void print_matrix(float *m, TensorDim mDims)
{
  for (int n=0; n< mDims.n; n++){ 
    for (int k = 0; k <mDims.c; k++) {
      for (int i = 0; i <mDims.h; i++){
        for (int j = 0; j <mDims.w; j++){
          int index =   ((n*mDims.h+i) *mDims.w +j) *mDims.c + k;
            cout<< m[index]<<" ";
        } cout<<endl;
      }cout<<endl;
    } 
  } 
}
 


// print a 4D Tensor in NCHW
void print_matrixNCHW(float *m, TensorDim mDims)
{
  for (int n=0; n< mDims.n; n++){ 
    for (int k = 0; k <mDims.c; k++) {
      for (int i = 0; i <mDims.h; i++){
        for (int j = 0; j <mDims.w; j++){
          int index =  n*mDims.c* mDims.w*mDims.h+ k *mDims.w*mDims.h + i * mDims.w + j;
            printf("%0.2f  ", m[index]);
        } cout<<endl;
      }cout<<endl;
    } 
  } 
}

 

// print a 4D Tensor in NCHW
void print_matrixNCHW2(float *m, TensorDim mDims)
{
  int n=0;
  int k = 0; 
      for (int j = 0; j <mDims.h; j++){
        for (int i = 0; i <mDims.w; i++){
          int index =  n*mDims.c* mDims.w*mDims.h+ k *mDims.w*mDims.h + j * mDims.w + i;
            printf("%0.2f  ", m[index]);
        } cout<<endl;
      }cout<<endl;
} 


bool TensorCompare(float *m1, float *m2, TensorDim dim) 
{
  bool ret = true;
  int N = TensorSize(dim);
  float mse = 0;
  for (int n = 0; n < dim.n; ++n) {
    for (int c = 0; c < dim.c; c++) {
      for (int h = 0; h < dim.h; ++h) {
        for (int w = 0; w < dim.w; w++) {
          int addr = w + dim.w * (h + dim.h * (c + dim.c * n));
          mse += pow(m1[addr] - m2[addr], 2);
        }
      }
    }
  } 
  mse = mse / N;
  if (mse > 1e-6) {
    ret = false;
  } 
  return ret;
}



int TensorSize(TensorDim dim) 
{
  return dim.n * dim.c * dim.h * dim.w;
}


void RandInitF32(float *p_data, int N) 
{
    int k;
    for (k = 0; k < N; k++) {
        float val = 5;// rand() % 10; 
        p_data[k] = val;
    }
}





