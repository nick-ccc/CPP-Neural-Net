#include "include/matrix.cuh"

template<typename T>
class Matrix {


    template <typename T>
    Matrix<T>::Matrix(size_t rows, size_t columns) {
        static_assert(std::is_same<T, int>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                    "Type not allowed. Use <int>, <float> or <double>.");
        
        this->rows      = rows;
        this->columns   = columns;
        this->my_array  = new T[rows*columns]; 
    }

    // template <class T>
    // Matrix<T>::Matrix(size_t height, size_t width, T min, T max) {
    //     static_assert(std::is_same<T, int>::value ||
    //                 std::is_same<T, float>::value ||
    //                 std::is_same<T, double>::value,
    //                 "Type not allowed. Use <int>, <float> or <double>.");
        
    //     this->rows      = rows;
    //     this->columns   = columns;
    //     this->array     = new T[rows*columns]; 
    //     allocateDeviceMemory();
    // }

    template <typename T>
    Matrix::~Matrix() {
        delete[] data;
        deallocateDeviceMemory();
    }

    template <typename T>
    void Matrix::allocateDeviceMemory() {
        cudaMalloc((void**)&matrix_device, rows * cols * sizeof(T));
    }

    template <typename T>
    void Matrix::deallocateDeviceMemory() {
        cudaFree(matrix_device);
    }

    template <typename T>
    void Matrix::copyToDevice() {
        cudaMemcpy(d_data, data, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    void Matrix::copyToHost() {
        cudaMemcpy(data, d_data, rows * cols * sizeof(flToat), cudaMemcpyDeviceToHost);
    }

    void Matrix::print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                std::cout << data[i * columns + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    
};
