/**
 * Matrix Linear Algebra Library - Fork from https://github.com/Vincouux/My-Linear-Algebra-Cuda/blob/master/src/Matrix/matrix.hpp
 */

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>
#include <iomanip>

#include "kernels.cuh"

template<typename T>
class Matrix {
    private: 
        size_t rows;
        size_t columns;
        std::shared_ptr<T[]> matrix;

    public:  
        // Constructors

        // Need a constructor that accepts the array we generate
        // when doing math 
        Matrix<T>();
        ~Matrix<T>();
        Matrix<T>(size_t rows, size_t columns);
        Matrix<T>(size_t height, size_t width, T min, T max);

        // Getters
        size_t getRows() const;
		size_t getColumns() const;
        T* getArrayPointer() const { return matrix.get();}

        

        // Matrix Operations 
        Matrix<T> add_gpu(const Matrix<T>& m) const;
        Matrix<T> add_cpu(const Matrix<T>& m) const;
        Matrix<T> sub_gpu(const Matrix<T>& m) const;
        Matrix<T> sub_cpu(const Matrix<T>& m) const;
        Matrix<T> multiply_gpu(const Matrix<T>& m) const;
        Matrix<T> multiply_cpu(const Matrix<T>& m) const;
        Matrix<T> divide_gpu(const Matrix<T>& m) const;
        Matrix<T> divide_cpu(const Matrix<T>& m) const;
        Matrix<T> inverse() const;
        Matrix<T> exponential() const;
        Matrix<T> power(unsigned n) const;
        Matrix<T> transpose();

};

#endif