 

#include <stdio.h>
#include "CONV_ref.h" 

bool is_a_ge_zero_and_a_lt_b_2(int a, int b) 
{
  return (unsigned int)a < (unsigned int)(b);
}

void RefConv2dF32(float *input, TensorDim inputDims, float *kernels, TensorDim kernelDims, float* output, TensorDim outputDims,  int pad,  int stride) 
{
  int imap_offset, omap_offset;

  for (int g = 0; g < inputDims.n; ++g) 
  {
    imap_offset = g * (inputDims.c / inputDims.n);
    omap_offset = g * (outputDims.c / inputDims.n);
    int s = 0;
    while (s < outputDims.c / inputDims.n) 
    {
        int in_row = -pad;
        for (int out_row = 0; out_row < outputDims.h; ++out_row) 
        {
          int in_col = -pad;
          for (int out_col = 0; out_col < outputDims.w; ++out_col) 
          {
            float sum = 0.0;
            for (int imap = 0; imap < inputDims.c / inputDims.n; ++imap) 
            {
              int in_addr_base = (imap_offset + imap) * inputDims.h + in_row;
              int wt_addr_base = ((omap_offset + s) * inputDims.c / inputDims.n + imap);
              for (int kr = 0; kr < kernelDims.w; ++kr) 
              {

                int wt_addr0 = (wt_addr_base * kernelDims.w + kr) * kernelDims.w;
                int in_addr0 = (in_addr_base + kr) * inputDims.w + in_col;

                for (int kc = 0; kc < kernelDims.h; ++kc) 
                {
                  if (is_a_ge_zero_and_a_lt_b_2(in_row + kr, inputDims.h)  && is_a_ge_zero_and_a_lt_b_2(in_col + kc, inputDims.w)) 
                  {
                    int in_addr = in_addr0 + kc;
                    int wt_addr = wt_addr0 + kc;
                    sum += kernels[wt_addr] * input[in_addr];
                  }
                }
              }
            } 
            int out_addr = ((omap_offset + s) * outputDims.h + out_row) * outputDims.w + out_col;
            output[out_addr] = sum;
            in_col += stride;
          }
          in_row += stride;
        }
        s++;
    }
  }
}
