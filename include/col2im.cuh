
// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);
 