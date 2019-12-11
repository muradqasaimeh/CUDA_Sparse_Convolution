#include "common.h"
#include "CONV_cuBLAS.cuh" 

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
								  const int height, const int width, const int ksize, const int pad,
								  const int stride, const int height_col, const int width_col,
								  float* data_col) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * ksize * ksize;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;

		float* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;

		for (int i = 0; i < ksize; ++i) 
		{
			for (int j = 0; j < ksize; ++j) 
			{
				int h = h_in + i;
				int w = w_in + j;
				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}

//__global__ void im2col_gpu_kernel(const int n, const float* data_im,
//    const int height, const int width, const int ksize, const int pad,
//    const int stride, const int height_col, const int width_col,
//    float* data_col) {
//  CUDA_KERNEL_LOOP(op_idx, n) {
//	int index = op_idx;
//    int w_out = index % width_col;
//
//    index /= width_col;
//    int h_out = index % height_col;
//    int channel_in = index / height_col;
//    int channel_out = channel_in * ksize * ksize;
//    int h_in = h_out * stride - pad;
//    int w_in = w_out * stride - pad;
//	
//    float* temp_col = data_col+ (channel_out * height_col + h_out) * width_col + w_out;
//    const float* temp_img = data_im + (channel_in * height + h_in) * width + w_in;
//	
//    for (int i = 0; i < ksize; ++i) {
//      for (int j = 0; j < ksize; ++j) {
//        int h = h_in + i;
//        int w = w_in + j;
//        *temp_col = (h >= 0 && w >= 0 && h < height && w < width) ?
//            temp_img[i * width + j] : 0;
//        temp_col += height_col * width_col;
//      }
//    }
//  }
//}
 
void im2col_gpu(const float* data_im, const int channels,
				const int height, const int width, const int ksize, const int pad,
				const int stride, float* data_col)
{
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		num_kernels, data_im, height, width, ksize, pad, stride, height_col,
		width_col, data_col);
	CUDA_POST_KERNEL_CHECK;
}

 /*
// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
								const int height, const int width, const int ksize, const int pad,
								const int stride, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
								 const int height, const int width, const int ksize, const int pad,
								 const int stride, double* data_col);
*/

// Helper function for using CUDA to add vectors in parallel.
//const float* data_im // raw data,
//const int channels // image channels
//const int height //image height
//const int width // image width
//const int ksize // kernel size
//const int pad // pad size
//const int stride // stride size
//const int height_col // output column height
//const int width_col // output column width
//float* data_col // outpu data

