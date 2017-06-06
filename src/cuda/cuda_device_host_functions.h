/*
* cuda_device_host_functions.h
*
*  Created on: Sep 18, 2015
*      Author: Fernando B Oliveira - fboliveira25@gmail.com
*
*  Description:
*
*/

#ifndef CUDA_DEVICE_HOST_FUNCTIONS_H_
#define CUDA_DEVICE_HOST_FUNCTIONS_H_

#include <cuda_runtime.h>

// A[i*m+j] (with i=0..n-1 and j=0..m-1).

__device__ __host__ inline int get_pos(int lines, int columns, int i, int j) {
    return (i*columns + j);
}

template<typename T> T get_value(T* data, int lines, int columns, int i, int j) {
    return data[get_pos(lines, columns, i, j)];
}

template float get_value(float* data, int lines, int columns, int i, int j);

#endif /* CUDA_DEVICE_HOST_FUNCTIONS_H_ */