#ifndef GPU_MAT_MULT
#define GPU_MAT_MULT

#include "matrix.cuh"

void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void tiled_mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3);

template<typename T>
void matrix_class_mult(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C);

#endif