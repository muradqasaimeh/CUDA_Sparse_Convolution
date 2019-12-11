 

// Explicit instantiation
void im2col_gpu(const float* data_im, const int channels,
				const int height, const int width, const int ksize, const int pad,
				const int stride, float* data_col); 