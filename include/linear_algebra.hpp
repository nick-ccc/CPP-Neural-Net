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
        std::vector<T> array;

    public:  
        // Constructors
        Matrix<T>();
        ~Matrix<T>();
        Matrix<T>(size_t height, size_t width);
        Matrix<T>(size_t height, size_t width, T min, T max);

        // Getters
        size_t getRows() const;
		size_t getColumns() const;
		T getDataFromIndices(size_t i, size_t j) const;

        // Matrix Operations 
        Matrix<T> add(const Matrix& m, bool gpu = true) const;
        Matrix<T> add(T m) const;
        Matrix<T> sub(const Matrix& m) const;
        Matrix<T> sub(T m) const;
        Matrix<T> dot(const Matrix<T>& m, bool gpu = true) const;
        Matrix<T> dot(T m) const;
        Matrix<T> multiply(const Matrix<T>& m) const;
        Matrix<T> divide(T m) const;
        Matrix<T> inverse(T m) const;
        Matrix<T> exponential() const;
        Matrix<T> power(unsigned n) const;
     
        Matrix transpose();

};

template <class T>
Matrix<T>::Matrix() {
    static_assert(std::is_same<T, int>::value ||
                  std::is_same<T, float>::value ||
                  std::is_same<T, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = 0;
    this->width = 0;
    this->array = std::vector<T>(0);
}

template <class T>
Matrix<T>::Matrix(size_t height, size_t width) {
    static_assert(std::is_same<T, int>::value ||
                  std::is_same<T, float>::value ||
                  std::is_same<T, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = height;
    this->width = width;
    this->array = std::vector<T>(this->height * this->width);
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            T n = -1 + rand() / T(RAND_MAX) * 2;
            this->setElementAt(i, j, n);
        }
    }
}

template <class T>
Matrix<T>::Matrix(size_t height, size_t width, T min, T max) {
    static_assert(std::is_same<T, int>::value ||
                  std::is_same<T, float>::value ||
                  std::is_same<T, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = height;
    this->width = width;
    this->array = std::vector<T>(this->height * this->width);
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            T n = min + (max - min) * (rand() / T(RAND_MAX));
            this->setElementAt(i, j, n);
        }
    }
}





#endif