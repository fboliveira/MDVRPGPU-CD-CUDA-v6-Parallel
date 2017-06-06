#ifndef CUDA_LOCAL_SEARCH_H_
#define CUDA_LOCAL_SEARCH_H_

#include <cuda_runtime.h>
//--#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_types.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include "../classes/Random.hpp"
#include "../classes/Route.hpp"

#include "../cuda/cuda_device_host_functions.h"

bool cudaOperateMoveDepotRouteM7(Route& u, int streamId);
bool cudaOperateMoveDepotRouteM8(Route& u, Route &v, int streamId);
bool cudaOperateMoveDepotRouteM9(Route& u, Route &v, int streamId);

//--bool cudaPerformeChange(Route& u, Route& v, int *routeU, int *routeV, int* bestChange, int move);
void cudaGetChanges(thrust::device_vector<float> bestCost, thrust::device_vector<int> bestI,
    thrust::device_vector<int> bestJ, int& bi, int& bj);
bool cudaPerformeChange(Route& u, Route& v, thrust::device_vector<int> routeU,
    thrust::device_vector<int> routeV, int bi, int bj, int move);

bool cudaLocalSearch(Route& u, Route& v, int move, int streamId);

#endif /* CUDA_LOCAL_SEARCH_H_ */
