#ifndef GPU_MAT_MULT
# define GPU_MAT_MULT

/**
 * TODO refactor this to work with the templated matrix class 
 * https://github.com/tgautam03/CUDA-C/blob/master/05_tiled_mat_mul/tiled_mat_mul_gpu.h
 * https://www.youtube.com/watch?v=Q3GgbfGTnVc
 */


void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void tiled_mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3);

#endif