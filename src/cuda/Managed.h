/*
 * Managed.h
 *
 *  Created on: Apr 16, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

// Adapted from https://github.com/parallel-forall/code-samples/tree/master/posts/unified-memory

#ifndef MANAGED_H_
#define MANAGED_H_

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Managed
{
public:
  void *operator new(size_t len) {
    void *ptr;
    //cout << "Len: " << len << endl;
    gpuErrchk( cudaMallocManaged(&ptr, len) );
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

#endif /* MANAGED_H_ */
