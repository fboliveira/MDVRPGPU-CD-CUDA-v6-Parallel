#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "cuda_types.h"

#include "../classes/MDVRPProblem.hpp"
#include "../classes/Random.hpp"

void cudaMutate(MDVRPProblem* problem, StrIndividual *subpop, StrIndividual *mngPopRes, int depot, int inds, int lambda);

#endif /* CUDA_KERNELS_H_ */