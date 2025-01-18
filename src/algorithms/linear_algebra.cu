#include "include/linear_algebra.cuh"

template<typename T>
class Matrix {
    
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
    Matrix<T>::Matrix(size_t rows, size_t columns) {
        static_assert(std::is_same<T, int>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                    "Type not allowed. Use <int>, <float> or <double>.");
        
        this->rows      = rows;
        this->columns   = columns;
        this->my_array  = new T[rows*columns]; 

        for (size_t i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                this->my_array[i][j] = 0;
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
        this->array = new T[rows*columns]; 

    }

    template <class T>
    Matrix<T> add_gpu(const Matrix<T>& m) const {

        int const i     = blockIdx.x * blockDim.x + threadIdx.x;
        int const j     = blockIdx.y * blockDim.y + threadIdx.y;
        int const row   = this.getRows()
        int const cols  = this.getRows()
        
        
        T* mat1         = this.getArray();
        T* mat2         = m.getArray();
        
        result = new T[rows*columns]; 

        if (i < rows && j < cols)
        {
            result[i * cols + j] = mat1[i * cols + j] + mat2[i * cols + j];

        }
    }

};
