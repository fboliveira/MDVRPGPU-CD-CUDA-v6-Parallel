#include "cuda_kernels.h"

__global__ void cudaKernelMutate(StrIndividual *subpop, StrIndividual *mngPopRes,
    int inds, int lambda, unsigned int seed) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < inds) {

        // Copy source to result, if lambda is not the first
        if (lambda > 0)
            for (int i = 0; i < subpop[idx].length; ++i)
                mngPopRes[idx].gene[i] = subpop[idx].gene[i];

        seed = seed + (idx * idx);

        //printf("Seed: %u\n", seed);

        //int* routes = new int[inds];

        //printf("Op: %d\n", subpop[idx].operations);

        // Source : http ://docs.nvidia.com/cuda/curand/index.html#ixzz3jxTjtPFu 
        //unsigned int seed = idx; 
        curandState s; // seed a random number generator 
        curand_init(seed, 0, 0, &s);

        int op = 0;
        int src = 0; //curand_poisson(&s, subpop[idx].length);
        int dest = 0;// curand_poisson(&s, subpop[idx].length);
        int aux;

        for (int i = 0; i < mngPopRes[idx].length; ++i) {

            if (op < mngPopRes[idx].operations) {
                src = curand_poisson(&s, mngPopRes[idx].length);
                dest = curand_poisson(&s, mngPopRes[idx].length);

                //printf("[%d] - Src: %d - Dest: %d\n", idx, src, dest);

                if (src < mngPopRes[idx].length && dest < mngPopRes[idx].length) {

                    aux = mngPopRes[idx].gene[src];
                    mngPopRes[idx].gene[src] = mngPopRes[idx].gene[dest];
                    mngPopRes[idx].gene[dest] = aux;

                    op++;

                }
            }
            else
                break;

        }

        __syncthreads();

        //delete[] routes;

    }

}

void cudaMutate(MDVRPProblem* problem, StrIndividual *subpop, StrIndividual *mngPopRes, 
    int depot, int inds, int lambda) {

    // Invoke kernel
    //int threadsPerBlock = 32;
    //int blocksPerGrid = (inds + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream = problem->getStream(depot);

    //cudaStreamAttachMemAsync(stream, subpop);
    //cudaStreamAttachMemAsync(stream, mngPopRes);
    //cudaStreamSynchronize(stream);

    //cudaDeviceSynchronize();
    cudaKernelMutate << <1, inds, 0, stream >> >(subpop, mngPopRes, inds, lambda, Random::generateSeed());

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    //cudaStreamDestroy(stream);

}