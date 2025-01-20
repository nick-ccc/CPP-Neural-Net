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
    public:  
        size_t rows;
        size_t columns;
        std::shared_ptr<T[]> matrix_host;
        std::shared_ptr<T[]> matrix_device;

        // Constructors
        ~Matrix<T>();
        Matrix<T>(size_t rows, size_t columns);
        // Matrix<T>(size_t height, size_t width, T min, T max);

        // // Getters
        // size_t getRows() const;
		// size_t getColumns() const;
        
        // // Assuming const is fine if we only change data in memory
        // T* getArrayPointer() const; 

        // Interface between host and device
        void allocateDeviceMemory();
        void deallocateDeviceMemory();
        void copyToDevice() const;
        void copyToHost() const;
        void print() const;
};

#endif