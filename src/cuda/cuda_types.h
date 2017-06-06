/*
 * cuda_types.h
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *
 */

#ifndef CUDA_TYPES_H_
#define CUDA_TYPES_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Managed.h"

// http://choorucode.com/2011/04/09/thrust-passing-device_vector-to-kernel/

// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
    T*  _array;
    int _size;
};

// Function to convert device_vector to structure
template < typename T >
KernelArray< T > convertToKernel(thrust::device_vector< T >& dVec)
{
    KernelArray< T > kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size = (int)dVec.size();

    return kArray;
}

// Template structure to pass to kernel
template < typename T >
struct KernelRoute : public KernelArray<T>
{

    int _depot;
    float _cost;
    int _demand;
    int _service;

};

// Function to convert device_vector to structure
template < typename T >
KernelRoute< T > convertRouteToKernel(thrust::device_vector< T >& dVec, int depot, float cost,
    int demand, int service)
{
    KernelRoute< T > kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size = (int)dVec.size();

    kArray._depot = depot;
    kArray._cost = cost;
    kArray._demand = demand;
    kArray._service = service;

    return kArray;
}

// Struct adapted from http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/

struct StrIndividual
{
    int length;
    int operations;
    int* gene;
    int* change;

public:

    void allocate(int l, cudaStream_t stream) {
        length = l;
        gpuErrchk(cudaMallocManaged(&gene, length * sizeof(int), cudaMemAttachHost));
        gpuErrchk(cudaMallocManaged(&change, length * sizeof(int), cudaMemAttachHost));

        cudaStreamAttachMemAsync(stream, gene);
        cudaStreamAttachMemAsync(stream, change);
        cudaStreamSynchronize(stream);
    }

    void deallocate() {
        cudaFree(gene);
        cudaFree(change);
    }

};

struct StrPop {

    int size;
    StrIndividual *individual;

    void allocate(int s) {
        size = s;
        cudaMallocManaged(&individual, size * sizeof(StrIndividual));
    }

};

class mngIndividual : public Managed {

    int length;
    int* gene;

public:
    // Unified memory copy constructor allows pass-by-value
    mngIndividual(int length) {
        this->length = length;
        cudaMallocManaged(&gene, length);
    }

    mngIndividual(const mngIndividual &ind) {
        length = ind.length;
        cudaMallocManaged(&gene, length);
        for (int i = 0; i < length; ++i)
            gene[i] = ind.gene[i];
    }



};

// Adapted from: https://thrust.github.io/doc/group__extrema.html#ga90f5158cab04adeb3f1b8b5e4acdbbcc

template < typename T >
struct min_element_and_greater_than
{
    T greater_than;
    min_element_and_greater_than(T _greater_than) :
        greater_than(_greater_than) {};

    __host__ __device__
        bool operator()(T lhs, T rhs)
    {
        return (lhs > greater_than) && (lhs < rhs);
    }
};

#endif /* CUDA_TYPES_H_ */
