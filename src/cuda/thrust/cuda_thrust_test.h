#ifndef CUDA_THRUST_TEST_H
#define CUDA_THRUST_TEST_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/find.h>

#include "../../classes/MDVRPProblem.hpp"
#include "../../classes/AlgorithmConfig.hpp"
#include "../../classes/Route.hpp"

#include "../cuda_types.h"

void cudaThrustTest(MDVRPProblem* problem, AlgorithmConfig* config);

#endif /* CUDA_THRUST_TEST_H */