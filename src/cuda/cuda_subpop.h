#ifndef CUDA_SUBPOP_H_
#define CUDA_SUBPOP_H_

#include <algorithm>

#include "../classes/Subpopulation.hpp"
#include "cuda_types.h"
#include "cuda_kernels.h"

class CudaSubpop {

public:

    static void evolve(MDVRPProblem* problem, StrIndividual *mngPop, StrIndividual *mngPopRes, int depot, int inds, int lambda);

private:
    static void mutate(MDVRPProblem* problem, StrIndividual *mngPop, StrIndividual *mngPopRes, int depot, int inds, int lambda);
    //static void localSearch(Subpopulation &subpop, IndividualsGroup &offsprings);
    //static void processTest(Subpopulation &subpop, IndividualsGroup &offsprings);

};

#endif /* CUDA_SUBPOP_H_ */