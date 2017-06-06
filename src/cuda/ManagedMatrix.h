/*
 * ManagedMatrix.h
 *
 *  Created on: Apr 15, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *
 */

#ifndef MANAGEDMATRIX_H_
#define MANAGEDMATRIX_H_

#include <iostream>
#include <cuda_runtime.h>

#include "../global.hpp"
#include "Managed.h"
#include "cuda_device_host_functions.h"

using namespace std;

template<class T>
class ManagedMatrix
{
private:

    // Attributes
    size_t lines;
    size_t columns;

    matrix_special<T> data;

    // Private methods

public:

    // Constructor
    ManagedMatrix();
    ManagedMatrix(size_t lines, size_t columns);

    // Destructor
    ~ManagedMatrix();

    size_t getLines() const;
    void setLines(size_t lines);

    size_t getColumns() const;
    void setColumns(size_t columns);

    T *getData();
    matrix_special<T> getMatrixData();

    __device__ __host__ inline T get(int line, int row);
    void set(int line, int row, T value);

    void init();
    void print();
    void print(bool asArray);


};

// Constructor
template<class T>
ManagedMatrix<T>::ManagedMatrix() {
    this->setLines(0);
    this->setColumns(0);
}

template<class T>
ManagedMatrix<T>::ManagedMatrix(size_t lines, size_t columns) {
    this->setLines(lines);
    this->setColumns(columns);
    init();
}

// Destructor
template<class T>
ManagedMatrix<T>::~ManagedMatrix() {

    cudaFree(data.data);

}
// Getters and Setters
template<class T>
size_t ManagedMatrix<T>::getLines() const {
    return lines;
}

template<class T>
void ManagedMatrix<T>::setLines(size_t lines) {
    this->lines = lines;
}

template<class T>
size_t ManagedMatrix<T>::getColumns() const {
    return columns;
}

template<class T>
void ManagedMatrix<T>::setColumns(size_t columns) {
    this->columns = columns;
}

template<class T>
T *ManagedMatrix<T>::getData() {
    return data.data;
}

template<class T>
matrix_special<T> ManagedMatrix<T>::getMatrixData() {
    return data;
}


// Methods
template<class T>
T ManagedMatrix<T>::get(int line, int row) { 
    return data.data[get_pos(this->getLines(), this->getColumns(), line, row)];
}

template<class T>
void ManagedMatrix<T>::set(int line, int row, T value) {
    data.data[get_pos(this->getLines(), this->getColumns(), line, row)] = value;
}

template<class T>
void ManagedMatrix<T>::init() {
    data.lines = lines;
    data.columns = columns;
    gpuErrchk(cudaMallocManaged(&data.data, lines * columns * sizeof(T *)));
}

template<class T>
void ManagedMatrix<T>::print() {
    for (int i = 0; i < this->getLines(); ++i) {
        for (int j = 0; j < this->getColumns(); ++j)
            cout << this->get(i, j) << "\t";
        cout << endl;
    }
}

template<class T>
void ManagedMatrix<T>::print(bool asArray) {
    for (int i = 0; i < this->getLines(); ++i) {
        for (int j = 0; j < this->getColumns(); ++j)
            cout << this->get(i, j) << ", ";
    }
}

#endif /* MANAGEDMATRIX_H_ */
