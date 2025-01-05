#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <vector>

template<typename T>
class Matrix {
    private: 
        int rows;
        int columns;
        int depth;
        T* data; 

    public:  
        Matrix();
        ~Matrix();
        Matrix(const Matrix&);
        Matrix& operator=(const Matrix&);
        Matrix(int rows, int columns);
        Matrix(int rows, int columns, int depth);

        int getRows();
		int getColumns();
		int getDepth();
		int getSize();
		T* getData();

        Matrix& operator+=(const Matrix&);
        Matrix& operator-=(const Matrix&);
        Matrix& operator*=(const Matrix&);
        Matrix& operator*=(double);
        Matrix& operator/=(double);
        Matrix  operator^(int);

        Matrix transpose();

        T getElement(int index);
};

template<typename T>
Matrix<T> operator+(const Matrix<T>&, const Matrix<T>&);

template<typename T>
Matrix<T> operator-(const Matrix<T>&, const Matrix<T>&);

template<typename T>
Matrix<T> operator*(const Matrix<T>&, const Matrix<T>&);

template<typename T>
Matrix<T> operator*(const Matrix<T>&, T);

template<typename T>
Matrix<T> operator*(T, const Matrix<T>&);

template<typename T>
Matrix<T> operator/(const Matrix<T>&, T);


#endif 