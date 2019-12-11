#include "CONV_cuDNN.cuh"

using namespace std;

void convolution_cudnn(float* h_input, TensorDim inputDims, float* h_kernels, TensorDim KernelDims, float* h_output)
{ 
  GpuTimer timer;
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
 
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, /*format=*/CUDNN_TENSOR_NCHW, /*dataType=*/CUDNN_DATA_FLOAT, /*batch_size=*/inputDims.n,
                                        /*channels=*/inputDims.c, /*image_height=*/inputDims.h, /*image_width=*/inputDims.w));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, /*format=*/CUDNN_TENSOR_NCHW, /*dataType=*/CUDNN_DATA_FLOAT, /*batch_size=*/inputDims.n,
                                        /*channels=*/KernelDims.n, /*image_height=*/inputDims.h, /*image_width=*/inputDims.w));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, /*dataType=*/CUDNN_DATA_FLOAT, /*format=*/CUDNN_TENSOR_NCHW, /*out_channels=*/KernelDims.n,
                                        /*in_channels=*/inputDims.c, /*kernel_height=*/KernelDims.h, /*kernel_width=*/KernelDims.w));


  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, /*pad_height=*/1, /*pad_width=*/1, /*vertical_stride=*/1, /*horizontal_stride=*/1,
                                           /*dilation_height=*/1, /*dilation_width=*/1, /*mode=*/CUDNN_CROSS_CORRELATION, /*computeType=*/CUDNN_DATA_FLOAT));


  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, /*memoryLimitInBytes=*/0, &convolution_algorithm));


  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
                                                   output_descriptor, convolution_algorithm, &workspace_bytes));
 
  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int in_tensor_bytes = inputDims.n * inputDims.h * inputDims.w * inputDims.c * sizeof(float);

  float* d_input{nullptr};
  cudaMalloc(&d_input, in_tensor_bytes);

  int out_tensor_bytes = inputDims.n * inputDims.h * inputDims.w  * KernelDims.n * sizeof(float);

  float* d_output{nullptr};
  cudaMalloc(&d_output, out_tensor_bytes);
  cudaMemset(d_output, 0, out_tensor_bytes);

  int kernel_bytes = KernelDims.n * KernelDims.h * KernelDims.w  * KernelDims.c * sizeof(float);

  float* d_kernel{nullptr};
  cudaMalloc(&d_kernel, kernel_bytes);
 

  timer.Start();

  cudaMemcpy(d_input, h_input , in_tensor_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernels, kernel_bytes, cudaMemcpyHostToDevice);
  timer.Stop(); 

  //printf("cudaMemcpyHostToDevice:\t %f msecs.\n", timer.Elapsed());

  // The Convolution
  const float alpha = 1, beta = 0;

  timer.Start();
  checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, convolution_descriptor,
                                   convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, d_output));
  timer.Stop(); 

  //printf("cuDNN Convolution:\t %f msecs.\n", timer.Elapsed());
 
  timer.Start();
  cudaMemcpy(h_output, d_output, out_tensor_bytes, cudaMemcpyDeviceToHost);
  timer.Stop();  

  //printf("cudaMemcpyDeviceToHost:\t %f msecs.\n", timer.Elapsed());


  //delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn); 
}


