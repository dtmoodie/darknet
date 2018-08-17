#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, uint64_t n);
void cuda_random(float *x_gpu, uint64_t n);
float cuda_compare(float *x_gpu, float *x, uint64_t n, char *s);
dim3 cuda_gridsize(uint64_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
