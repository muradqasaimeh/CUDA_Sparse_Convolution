

void NCHW2HWNC(const float *nchw_data, int N, int C, int H, int W, float *hwnc_data);
void NCHW2HWCN(const float *nchw_data, int N, int C, int H, int W, float *hwcn_data);

void NCHW2NHWC(const float *nchw_data, int N, int C, int H, int W, float *nhwc_data);
void NCHW2CHWN(const float *nchw_data, int N, int C, int H, int W, float *chwn_data);
void NHWC2NCHW(const float *nhwc_data, int N, int C, int H, int W, float *nchw_data);
