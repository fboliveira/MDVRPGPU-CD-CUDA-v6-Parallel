/*
 * ManagedArray.h
 *
 *  Created on: Apr 15, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *
 */

#ifndef MANAGEDARRAY_H_
#define MANAGEDARRAY_H_

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

template<class T>
class ManagedArray
{
private:

    // Attributes
    size_t cols;

    T *data;

    // Private methods

public:

    // Constructor
    ManagedArray();
    ManagedArray(size_t cols);

    // Destructor
    ~ManagedArray();

    size_t getCols() const;
    void setCols(size_t cols);

    T *getData();

    T get(int col);
    void set(int col, T value);

    void init();
    void print();

    T operator[](std::size_t index) {
        return data[index];
    }

};

// Constructor
template<class T>
ManagedArray<T>::ManagedArray() {
    this->setCols(0);
}

template<class T>
ManagedArray<T>::ManagedArray(size_t cols) {
    this->setCols(cols);
    init();
}

// Destructor
template<class T>
ManagedArray<T>::~ManagedArray() {
    cudaFree(data);
}
// Getters and Setters
template<class T>
size_t ManagedArray<T>::getCols() const {
    return cols;
}

template<class T>
void ManagedArray<T>::setCols(size_t cols) {
    this->cols = cols;
}

template<class T>
T *ManagedArray<T>::getData() {
    return data;
}

// Methods
template<class T>
T ManagedArray<T>::get(int col) {
    return data[col];
}

template<class T>
void ManagedArray<T>::set(int col, T value) {
    data[col] = value;
}

template<class T>
void ManagedArray<T>::init() {
    cudaMallocManaged(&data, cols * sizeof(T *));
}

template<class T>
void ManagedArray<T>::print() {

    for (int i = 0; i < this->getCols(); ++i)
        cout << this->get(i) << "\t";

    cout << endl;
}

#endif /* MANAGEDARRAY_H_ */
