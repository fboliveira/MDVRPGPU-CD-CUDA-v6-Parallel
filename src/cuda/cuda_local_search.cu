#include "cuda_local_search.h"

// http://stackoverflow.com/questions/1343890/rounding-number-to-2-decimal-places-in-c
__device__ float cuda_scaled_float(float value) {
    value = rintf(value * 100) / 100;
    return value;
}

__device__ bool cuda_isbetter_scaled(float newCost, float currentCost) {

    if (cuda_scaled_float(newCost) >= 0 && cuda_scaled_float(newCost) < cuda_scaled_float(currentCost))
        return true;
    else
        return false;

}

__device__ bool cuda_isbetter(float newCost, float currentCost) {

    if (cuda_scaled_float(newCost) >= 0 && (currentCost - newCost) > 0.01f)
        return true;
    else
        return false;

}

__device__ float cuda_customer_cost_before(int before, int customer, int depot,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances) {

    float cost = 0.0f;

    //Before -> Customer:
    // If it is in the front of
    if (before < 0) {
        // D->C
        if (customer <= 0)
            printf("c <= 0");
        else
            cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines, mngDepotDistances.columns, depot, customer - 1)];
    }
    else {
        // before -> C
        if (customer <= 0 || before < 0)
            printf("c: %d \tb:%d\n", customer, before);
        else
            cost += mngCustomerDistances.data[get_pos(mngCustomerDistances.lines, mngCustomerDistances.columns,
            before - 1, customer - 1)];
    }

    return cost;

}

__device__ float cuda_customer_cost_after(int customer, int after, int depot,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances) {

    float cost = 0.0f;

    // Customer -> After:
    // Last one
    if (after < 0) {
        // C->D
        cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines, mngDepotDistances.columns, depot, customer - 1)];
    }
    else { // Anywhere...
        // C->C+1
        cost += mngCustomerDistances.data[get_pos(mngCustomerDistances.lines, mngCustomerDistances.columns,
            customer - 1, after - 1)];
    }

    return cost;

}

__device__ float cuda_customer_cost(int before, int customer, int after, int depot,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances) {

    float cost = 0.0f;

    //Before -> Customer -> After:
    cost += cuda_customer_cost_before(before, customer, depot, mngCustomerDistances, mngDepotDistances);
    cost += cuda_customer_cost_after(customer, after, depot, mngCustomerDistances, mngDepotDistances);

    return cost;

}

__device__ int cuda_get_customer_before(int* route, int idx) {

    int before;

    // Before
    if (idx <= 0)
        before = -1;
    else
        before = route[idx - 1];

    return before;

}

__device__ int cuda_get_customer_after(int* route, int size, int idx) {

    int after;

    // After
    if ((idx + 1) >= size)
        after = -1;
    else
        after = route[idx + 1];

    return after;

}

__device__ void cuda_find_bestChange(int* bestChange, float* bestCost, int sizeU) {

    int bi = -1;
    int bcust = -1;
    float bcost = FLT_MAX;

    for (int i = 0; i < sizeU; ++i) {

        //printf("C: %d -> %d [%.2f] \n", i, bestChange[i], bestCost[i]);

        if (bestChange[i] >= 0)
            if (cuda_isbetter(bestCost[i], bcost)) {
                bi = i;
                bcost = bestCost[i];
                bcust = bestChange[i];
            }

    }

    bestChange[0] = bi;
    bestChange[1] = bcust;

}
// Calculate route cost between D->[first..last]->D
__device__ float cuda_calculate_route_cost(int *route, int first, int last, int depot, int *demand, int *service,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService) {

    float cost = 0.0f;
    *demand = 0;
    *service = 0;

    // From depot to first
    cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines,
        mngDepotDistances.columns, depot, route[first] - 1)];
    // From depot to last
    cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines,
        mngDepotDistances.columns, depot, route[last] - 1)];

    int customer = 0;

    //printf("cuda_calculate_route_cost: %d to %d => ", first, last);

    for (int i = first; i <= last; ++i) {

        customer = route[i] - 1;
        //printf("%d", customer + 1);

        if (customer >= 0) {

            *demand += mngDemand[customer];
            *service += mngService[customer];

            if (i + 1 <= last) {
                int j = route[i + 1] - 1;
                //printf(" - (%d)\t", j + 1);

                if (j >= 0) {
                    cost += mngCustomerDistances.data[get_pos(mngCustomerDistances.lines, mngCustomerDistances.columns,
                        customer, j)];
                }

            }
        }
    }

    //printf("\n\n");

    return cost;

}

__host__ __device__ void cuda_2opt_swap(int *route, int size, int *newRoute, int i, int k) {

    int p = 0;

    // https://en.wikipedia.org/wiki/2-opt

    //printf("cuda_2opt_swap => i = %d - k = %d => ", i, k);

    // 1. take route[1] to route[i-1] and add them in order to new_route
    for (int c = 0; c <= i - 1; ++c) {
        newRoute[p] = route[c];
        //printf("%d\t", newRoute[p]);
        p++;
    }

    // 2. take route[i] to route[k] and add them in reverse order to new_route
    for (int c = k; c >= i; --c) {
        newRoute[p] = route[c];
        //printf("%d\t", newRoute[p]);
        p++;
    }

    // 3. take route[k+1] to end and add them in order to new_route
    for (int c = k + 1; c < size; ++c) {
        newRoute[p] = route[c];
        //printf("%d\t", newRoute[p]);
        p++;
    }

    //printf("\n\n");

}

__host__ __device__ void cuda_2opt_swap_star(int *route, int size, int *newRoute, int i, int k, int* delim) {

    int p = 0;

    *delim = -1;

    // 2-opt*
    // Adapted from: https://en.wikipedia.org/wiki/2-opt

    //printf("cuda_2opt_swap_star => i = %d - k = %d => ", i, k);

    // 1. take route[1] to route[i-1] and add them in order to new_route
    for (int c = 0; c <= i - 1; ++c) {
        newRoute[p] = route[c];
        
        if (route[c] == -1)
            *delim = p;

        //printf("%d\t", newRoute[p]);
        p++;
    }

    // 2. take route[k+1] to end and add them in order to new_route
    for (int c = k + 1; c < size; ++c) {
        newRoute[p] = route[c];
        if (route[c] == -1)
            *delim = p;
        //printf("%d\t", newRoute[p]);
        p++;
    }

    // 3. take route[i] to route[k] and add them in order to new_route
    for (int c = i; c <= k; ++c) {
        newRoute[p] = route[c];
        if (route[c] == -1)
            *delim = p;
        //printf("%d\t", newRoute[p]);
        p++;
    }

    //printf("\n\n");

}

// (M1) If u is a customer visit, remove u and place it after v;
__global__ void cudaKernelLS_M1(KernelRoute<int> u, KernelRoute<int> v,
    KernelArray<int> bestChange, KernelArray<float> bestCost,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < u._size) {

        int beforeU, customerU, afterU, beforeV, customerV, afterV;
        //float costU, costV;

        customerU = u._array[idx];
        beforeU = cuda_get_customer_before(u._array, idx);
        afterU = cuda_get_customer_after(u._array, u._size, idx);

        // remove: b -> u -> a
        // add: b -> a
        float costRemoveU = 0.0f;

        if (u._size > 1) {
            costRemoveU = u._cost - cuda_customer_cost(beforeU, customerU, afterU,
                u._depot, mngCustomerDistances, mngDepotDistances);

            if (afterU < 0)
                costRemoveU += cuda_customer_cost_after(beforeU, afterU,
                u._depot, mngCustomerDistances, mngDepotDistances);
            else
                costRemoveU += cuda_customer_cost_before(beforeU, afterU,
                u._depot, mngCustomerDistances, mngDepotDistances);
        }

        float costInsertUV;
        int serviceInsertUV;

        bestChange._array[idx] = -1;
        bestCost._array[idx] = FLT_MAX;

        // Check demand for V
        if (v._demand + mngDemand[customerU - 1] <= capacity) {

            serviceInsertUV = v._service + mngService[customerU - 1];

            //[d] -> ... [d]
            for (int i = 0; i <= v._size; ++i) {

                // After last position
                if (i == v._size) {
                    customerV = u._array[i - 1];
                    beforeV = customerV;
                    afterV = -1;

                    // Evaluate insert u (idx) on v (i);
                    //[i-1] -> [i] TO [i-1] -> [u] -> [i]
                    costInsertUV = v._cost - cuda_customer_cost_after(customerV, afterV,
                        v._depot, mngCustomerDistances, mngDepotDistances) +
                        cuda_customer_cost(customerV, customerU, afterV,
                        v._depot, mngCustomerDistances, mngDepotDistances);
                }
                else {
                    customerV = v._array[i];
                    beforeV = cuda_get_customer_before(v._array, i);
                    afterV = cuda_get_customer_after(v._array, v._size, i);

                    // Evaluate insert u (idx) on v (i);
                    //[i-1] -> [i] TO [i-1] -> [u] -> [i]
                    costInsertUV = v._cost - cuda_customer_cost_before(beforeV, customerV,
                        v._depot, mngCustomerDistances, mngDepotDistances) +
                        cuda_customer_cost(beforeV, customerU, customerV,
                        v._depot, mngCustomerDistances, mngDepotDistances);
                }

                // Check duration for V
                if (costInsertUV + serviceInsertUV > duration)
                    continue;

                // Check improvement
                if (cuda_isbetter(costRemoveU + costInsertUV, u._cost + v._cost))
                    // Check best change
                    if (cuda_isbetter(costRemoveU + costInsertUV, bestCost._array[idx])) {
                        bestCost._array[idx] = costRemoveU + costInsertUV;
                        bestChange._array[idx] = i;
                    }
            }
        }
    }

    __syncthreads();

    // Find best change
    if (idx == 0) {
        cuda_find_bestChange(bestChange._array, bestCost._array, u._size);
    }


}

// (M4) If u and v are customer visits, swap u and v;
__global__ void cudaKernelLS_M4(KernelRoute<int> u, KernelRoute<int> v,
    KernelArray<int> bestChange, KernelArray<float> bestCost,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //if (idx == 0) {
    //    for (int i = 0; i < sizeU; ++i)
    //        printf("%d\t", routeU[i]);

    //    printf("\n");
    //}

    //__syncthreads();

    if (idx < u._size) {

        //if (idx >= 5)
        //    printf("Idx = %d\n", idx);

        int beforeU, customerU, afterU, beforeV, customerV, afterV;
        float costU, costV;

        customerU = u._array[idx];
        beforeU = cuda_get_customer_before(u._array, idx);
        afterU = cuda_get_customer_after(u._array, u._size, idx);

        float costRemoveU = u._cost - cuda_customer_cost(beforeU, customerU, afterU,
            u._depot, mngCustomerDistances, mngDepotDistances);
        int demandRemoveU = u._demand - mngDemand[customerU - 1];
        int serviceRemoveU = u._service - mngService[customerU - 1];

        float costRemoveV;
        int demandRemoveV;
        int serviceRemoveV;

        bestChange._array[idx] = -1;
        bestCost._array[idx] = FLT_MAX;

        for (int i = 0; i < v._size; ++i) {

            customerV = v._array[i];
            beforeV = cuda_get_customer_before(v._array, i);
            afterV = cuda_get_customer_after(v._array, v._size, i);

            // Check demand for U
            if (demandRemoveU + mngDemand[customerV - 1] > capacity)
                continue;

            // Evaluate swap between u (idx) and v (i);
            costRemoveV = v._cost - cuda_customer_cost(beforeV, customerV, afterV,
                v._depot, mngCustomerDistances, mngDepotDistances);

            demandRemoveV = v._demand - mngDemand[customerV - 1];
            serviceRemoveV = v._service - mngService[customerV - 1];

            // U + v
            costU = costRemoveU + cuda_customer_cost(beforeU, customerV, afterU, u._depot,
                mngCustomerDistances, mngDepotDistances);

            // Check duration for U
            if (costU + serviceRemoveU + mngService[customerV - 1] > duration)
                continue;

            // V + u
            costV = costRemoveV + cuda_customer_cost(beforeV, customerU, afterV, v._depot,
                mngCustomerDistances, mngDepotDistances);
            demandRemoveV += mngDemand[customerU - 1];
            serviceRemoveV += mngService[customerU - 1];

            // Check duration for V
            if (costV + serviceRemoveV > duration)
                continue;

            // Check demand for V
            if (demandRemoveV > capacity)
                continue;

            // Check improvement
            if (cuda_isbetter(costU + costV, u._cost + v._cost))
                // Check best change
                if (cuda_isbetter(costU + costV, bestCost._array[idx])) {
                    bestCost._array[idx] = costU + costV;
                    bestChange._array[idx] = i;
                }
        }
    }

    __syncthreads();

    // Find best change
    if (idx == 0) {
        cuda_find_bestChange(bestChange._array, bestCost._array, u._size);
    }


}

// (M7) If r(u)=r(v),   replace (u,x) and (v,y) by (u,v) and (x,y);
__global__ void cudaKernelLS_M7(KernelRoute<int> u,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration,
    KernelArray<int> bestI, KernelArray<int> bestK, KernelArray<float> bestCost) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int bi = -1;
    int bk = -1;

    float bcost = u._cost;
    float cost;

    bestCost._array[idx] = FLT_MAX;

    if (idx < u._size - 1) {

        int *newRoute = new int[u._size];

        for (int k = idx + 1; k < u._size; ++k) {

            cuda_2opt_swap(u._array, u._size, newRoute, idx, k);
            cost = cuda_calculate_route_cost(newRoute, 0, u._size - 1, u._depot, &u._demand, &u._service,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            // Check service duration
            if (cost + u._service <= duration)
                if (cuda_scaled_float(cost) < cuda_scaled_float(bcost)) {
                    bi = idx;
                    bk = k;
                    bcost = cost;
                }
        }

        bestI._array[idx] = bi;
        bestK._array[idx] = bk;
        bestCost._array[idx] = bcost;

        delete[] newRoute;

    }

}

__global__ void cudaKernelLS_M8(KernelRoute<int> u, KernelRoute<int> v, KernelArray<int> uv,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration,
    KernelArray<int> bestI, KernelArray<int> bestK, KernelArray<float> bestCost) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int bi = -1;
    int bk = -1;

    float bcost = u._cost + v._cost;
    float costU, costV;

    bestCost._array[idx] = FLT_MAX;

    if (idx < u._size) { //  

        int *newRoute = new int[uv._size];
        int us, ud, vs, vd, p;

        for (int k = u._size + 1; k < uv._size; ++k) {

            cuda_2opt_swap(uv._array, uv._size, newRoute, idx, k);

            // new position of delimiter after swap
            // i + (k - u._size)
            // A-B-C-D-E-0-F-G-H-I
            p = idx + (k - u._size);
            //printf("p = %d\n", p);

            costU = cuda_calculate_route_cost(newRoute, 0,  p - 1, u._depot, &ud, &us,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            costV = cuda_calculate_route_cost(newRoute, p + 1, uv._size - 1, v._depot, &vd, &vs,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            // Check service duration and capacity
            if (costU + us <= duration && ud <= capacity)
                if (costV + vs <= duration && vd <= capacity)
                    if (cuda_scaled_float(costU + costV) < cuda_scaled_float(bcost)) {
                        bi = idx;
                        bk = k;
                        bcost = costU + costV;
                    }
        }

        bestI._array[idx] = bi;
        bestK._array[idx] = bk;
        bestCost._array[idx] = bcost;

        delete[] newRoute;

    }

    //__syncthreads();

    //if (idx == 0) {
    //    for (int i = 0; i < uv._size; ++i)
    //        printf("i = %d\tbi = %d\tbk = %d\tbc = %f\n", i, bestI._array[i], bestK._array[i], bestCost._array[i]);
    //}
}

__global__ void cudaKernelLS_M9(KernelRoute<int> u, KernelRoute<int> v, KernelArray<int> uv,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration,
    KernelArray<int> bestI, KernelArray<int> bestK, KernelArray<float> bestCost) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int bi = -1;
    int bk = -1;

    float bcost = u._cost + v._cost;
    float costU, costV;

    bestCost._array[idx] = FLT_MAX;

    if (idx < u._size) { //;

        int *newRoute = new int[uv._size];
        int us, ud, vs, vd, p;

        for (int k = u._size + 1; k < uv._size; ++k) {

            // new position of delimiter after swap
            cuda_2opt_swap_star(uv._array, uv._size, newRoute, idx, k, &p);
            //printf("p = %d\n", p);

            costU = cuda_calculate_route_cost(newRoute, 0, p - 1, u._depot, &ud, &us,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            costV = cuda_calculate_route_cost(newRoute, p + 1, uv._size - 1, v._depot, &vd, &vs,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            // Check service duration and capacity
            if (costU + us <= duration && ud <= capacity)
                if (costV + vs <= duration && vd <= capacity)
                    if (cuda_scaled_float(costU + costV) < cuda_scaled_float(bcost)) {
                        bi = idx;
                        bk = k;
                        bcost = costU + costV;
                    }
        }

        bestI._array[idx] = bi;
        bestK._array[idx] = bk;
        bestCost._array[idx] = bcost;

        delete[] newRoute;

    }

    //__syncthreads();

    //if (idx == 0) {
    //    for (int i = 0; i < uv._size; ++i)
    //        printf("i = %d\tbi = %d\tbk = %d\tbc = %f\n", i, bestI._array[i], bestK._array[i], bestCost._array[i]);
    //}
}

bool cudaOperateMoveDepotRouteM7(Route& u, int streamId) {

    bool result = false;

    int size = (int)u.getTour().size();

    if (size < 4)
        return result;

    //cudaStream_t stream = u.getProblem()->getStream(streamId);

    thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());

    thrust::device_vector<int> bestI(size);
    thrust::device_vector<int> bestK(size);
    thrust::device_vector<float> bestCost(size);

    // , 0, stream
    cudaKernelLS_M7 << <1, size >> >(convertRouteToKernel(d_u, u.getDepot(), u.getCost(), u.getDemand(), u.getServiceTime()),
        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
        u.getProblem()->getCapacity(), u.getProblem()->getDuration(),
        convertToKernel(bestI), convertToKernel(bestK), convertToKernel(bestCost));
    
    //cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    // https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialWhenSomethingGoesWrong
    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("cudaOperateMoveDepotRouteM7: CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    int bi = -1;
    int bk = -1;

    cudaGetChanges(bestCost, bestI, bestK, bi, bk);

    // There are two different positions
    if (bi >= 0 && bk > bi) {
        int *newRote = new int[size];
        thrust::host_vector<int> h_u = d_u;
        cuda_2opt_swap(thrust::raw_pointer_cast(&h_u[0]), size, newRote, bi, bk);

        u.vectorToRoute(newRote, size);
        result = true;
        delete[] newRote;
    }

    //cudaStreamDestroy(stream);

    return result;

}

// (M8) If r(u) != r(v),replace (u,x) and (v,y) by (u,v) and (x,y);
bool cudaOperateMoveDepotRouteM8(Route& u, Route &v, int streamId) {

    bool result = false;

    if (u.getTour().size() < 2)
        return result;

    if (v.getTour().size() < 2)
        return result;

    //cudaStream_t stream = u.getProblem()->getStream(streamId);

    thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());
    thrust::device_vector<int> d_v(v.getTour().begin(), v.getTour().end());

    // Merge U and V
    thrust::device_vector<int> d_uv(u.getTour().begin(), u.getTour().end());
    // Insert route delimiter
    d_uv.push_back(-1);
    // Insert V
    d_uv.insert(d_uv.end(), v.getTour().begin(), v.getTour().end());

    int sizeU = (int)u.getTour().size();
    int sizeV = (int)v.getTour().size();
    int size = sizeU + sizeV + 1;

    thrust::device_vector<int> bestI(size);
    thrust::device_vector<int> bestK(size);
    thrust::device_vector<float> bestCost(size);

    // , 0, stream
    cudaKernelLS_M8 << <1, (int)u.getTour().size() >> >(convertRouteToKernel(d_u, u.getDepot(), u.getCost(), u.getDemand(), u.getServiceTime()),
        convertRouteToKernel(d_v, v.getDepot(), v.getCost(), v.getDemand(), v.getServiceTime()),
        convertToKernel(d_uv),
        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
        u.getProblem()->getCapacity(), u.getProblem()->getDuration(),
        convertToKernel(bestI), convertToKernel(bestK), convertToKernel(bestCost));

    //cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    int bi = -1;
    int bk = -1;

    cudaGetChanges(bestCost, bestI, bestK, bi, bk);

    // There are two different positions
    if (bi >= 0 && bk > bi) {
        int *newRote = new int[size];
        thrust::host_vector<int> h_u = d_uv;
        cuda_2opt_swap(thrust::raw_pointer_cast(&h_u[0]), size, newRote, bi, bk);

        //thrust::host_vector<int>::iterator iter =
        //    thrust::find(h_u.begin(), h_u.end(), -1);

        //// print Y
        //thrust::copy(h_u.begin(), h_u.end(), std::ostream_iterator<int>(std::cout, "\t"));

        //unsigned int position = (iter - h_u.begin());

        //cout << d_uv[position] << endl;

        int p = bi + (bk - sizeU);

        //Util::print(newRote, size);
        //Util::print(newRote, 0, p - 1);
        //Util::print(newRote, p + 1, size - 1);

        u.vectorToRoute(newRote, 0, p - 1);
        v.vectorToRoute(newRote, p + 1, size - 1);

        //u.printSolution();
        //v.printSolution();

        result = true;
        delete[] newRote;
    }

    return result;

}

// (M9) If r(u) != r(v),replace (u,x) and (v,y) by (u,y) and (x,v).
bool cudaOperateMoveDepotRouteM9(Route& u, Route &v, int streamId) {

    bool result = false;

    if (u.getTour().size() < 2)
        return result;

    if (v.getTour().size() < 2)
        return result;

    //cudaStream_t stream = u.getProblem()->getStream(streamId);

    thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());
    thrust::device_vector<int> d_v(v.getTour().begin(), v.getTour().end());

    // Merge U and V
    thrust::device_vector<int> d_uv(u.getTour().begin(), u.getTour().end());
    // Insert route delimiter
    d_uv.push_back(-1);
    // Insert V
    d_uv.insert(d_uv.end(), v.getTour().begin(), v.getTour().end());

    int sizeU = (int)u.getTour().size();
    int sizeV = (int)v.getTour().size();
    int size = sizeU + sizeV + 1;

    thrust::device_vector<int> bestI(size);
    thrust::device_vector<int> bestK(size);
    thrust::device_vector<float> bestCost(size);

    //, 0, stream
    cudaKernelLS_M9 << <1, (int)u.getTour().size() >> >(convertRouteToKernel(d_u, u.getDepot(), u.getCost(), u.getDemand(), u.getServiceTime()),
        convertRouteToKernel(d_v, v.getDepot(), v.getCost(), v.getDemand(), v.getServiceTime()),
        convertToKernel(d_uv),
        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
        u.getProblem()->getCapacity(), u.getProblem()->getDuration(),
        convertToKernel(bestI), convertToKernel(bestK), convertToKernel(bestCost));
    
    //cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    int bi = -1;
    int bk = -1;

    cudaGetChanges(bestCost, bestI, bestK, bi, bk);

    // There are two different positions
    if (bi >= 0 && bk > bi) {
        int *newRote = new int[size];
        int p;
        thrust::host_vector<int> h_u = d_uv;
        cuda_2opt_swap_star(thrust::raw_pointer_cast(&h_u[0]), size, newRote, bi, bk, &p);

        //thrust::host_vector<int>::iterator iter =
        //    thrust::find(h_u.begin(), h_u.end(), -1);

        //// print Y
        //thrust::copy(h_u.begin(), h_u.end(), std::ostream_iterator<int>(std::cout, "\t"));

        //unsigned int position = (iter - h_u.begin());

        //cout << d_uv[position] << endl;
        //cout << newRote[p] << endl;

        //Util::print(newRote, size);
        //Util::print(newRote, 0, p - 1);
        //Util::print(newRote, p + 1, size - 1);

        u.vectorToRoute(newRote, 0, p - 1);
        v.vectorToRoute(newRote, p + 1, size - 1);

        //u.printSolution();
        //v.printSolution();

        result = true;
        delete[] newRote;
    }

    return result;

}

void cudaGetChanges(thrust::device_vector<float> bestCost, thrust::device_vector<int> bestI,
    thrust::device_vector<int> bestJ, int& bi, int& bj) {

    // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
    thrust::device_vector<float>::iterator iter =
        thrust::min_element(thrust::device, bestCost.begin(), bestCost.end(), min_element_and_greater_than<float>(0));

    unsigned int position = iter - bestCost.begin();
    bi = bestI[position];
    bj = bestJ[position];

}

bool cudaPerformeChange_OLD(Route& u, Route& v, thrust::device_vector<int> routeU,
    thrust::device_vector<int> routeV, thrust::device_vector<int> bestChange, int move) {

    bool result = false;

    int bi;
    int bcust;

    bi = bestChange[0];
    bcust = bestChange[1];

    if (bi >= 0) {

        int sizeU = u.getTour().size();
        int sizeV = v.getTour().size();

        result = true;

        switch (move) {

        case 1:

            // Insert u on V
            // Remove from U
            //if (u.getTour().size() == 1)
            //    u.printSolution();

            u.remove(routeU[bi]);
            // Insert on V

            if (bcust >= v.getTour().size()) {
                v.addAtBack(routeU[bi]);
                //--Util::insert(routeV, sizeV, sizeV, routeU[bi]);
            }
            else {
                bcust--;
                if (bcust < 0) {
                    v.addAtFront(routeU[bi]);
                    //--Util::insert(routeV, sizeV, 0, routeU[bi]);
                }
                else {
                    v.addAfterPrevious(routeV[bcust], routeU[bi]);
                    //--Util::insert(routeV, sizeV, bcust + 1, routeU[bi]);
                }
            }

            //--Util::remove(routeU, sizeU, bi);

            //u.printSolution();
            //Util::print(routeU, sizeU - 1);
            //v.printSolution();
            //Util::print(routeV, sizeV + 1);

            break;

        case 4:
            // Swap u and v
            u.changeCustomer(u.find(routeU[bi]), routeV[bcust]);
            v.changeCustomer(v.find(routeV[bcust]), routeU[bi]);
            //--Util::change(routeU, bi, routeV, bcust);
            break;

        }

    }

    return result;

}

bool cudaPerformeChange(Route& u, Route& v, thrust::device_vector<int> routeU,
    thrust::device_vector<int> routeV, int bi, int bj, int move) {

    bool result = false;

    if (bi >= 0) {

        int sizeU = u.getTour().size();
        int sizeV = v.getTour().size();

        result = true;

        switch (move) {

        case 1:

            // Insert u on V
            // Remove from U
            //if (u.getTour().size() == 1)
            //    u.printSolution();

            u.remove(routeU[bi]);
            // Insert on V

            if (bj >= v.getTour().size()) {
                v.addAtBack(routeU[bi]);
                //--Util::insert(routeV, sizeV, sizeV, routeU[bi]);
            }
            else {
                bj--;
                if (bj < 0) {
                    v.addAtFront(routeU[bi]);
                    //--Util::insert(routeV, sizeV, 0, routeU[bi]);
                }
                else {
                    v.addAfterPrevious(routeV[bj], routeU[bi]);
                    //--Util::insert(routeV, sizeV, bcust + 1, routeU[bi]);
                }
            }

            //--Util::remove(routeU, sizeU, bi);

            //u.printSolution();
            //Util::print(routeU, sizeU - 1);
            //v.printSolution();
            //Util::print(routeV, sizeV + 1);

            break;

        case 4:
            // Swap u and v
            u.changeCustomer(u.find(routeU[bi]), routeV[bj]);
            v.changeCustomer(v.find(routeV[bj]), routeU[bi]);
            //--Util::change(routeU, bi, routeV, bcust);
            break;

        }

    }

    return result;

}

bool cudaLocalSearch_OLD(Route& u, Route& v, int move) {

    bool result = false;
    bool improved = false;

    //if (move == 7)
    //    return cudaOperateMoveDepotRouteM7(u, 0);

    int *routeU, *routeV, *bestChange;
    float* bestCost;

    //cout << "Start LS: " << move << " - U: " << u.getDepot() << " - V: " << v.getDepot() << endl;

    // http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    cudaStream_t stream;// = u.getProblem()->getStream(10 + move);
    cudaStreamCreate(&stream);

    size_t dataSize = (u.getTour().size() + v.getTour().size()) * sizeof(int);

    // size equals (u+v) to allow insert values on both routes
    gpuErrchk(cudaMallocManaged(&routeU, dataSize, cudaMemAttachHost));
    gpuErrchk(cudaMallocManaged(&bestChange, dataSize, cudaMemAttachHost));
    gpuErrchk(cudaMallocManaged(&bestCost, dataSize, cudaMemAttachHost));

    gpuErrchk(cudaMallocManaged(&routeV, dataSize, cudaMemAttachHost));

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-coherency-hd
    cudaStreamAttachMemAsync(stream, routeU);
    cudaStreamAttachMemAsync(stream, bestChange);
    cudaStreamAttachMemAsync(stream, bestCost);
    cudaStreamAttachMemAsync(stream, routeV);
    cudaStreamSynchronize(stream);

    cudaDeviceSynchronize();

    u.routeToVector(routeU);
    v.routeToVector(routeV);

    //do {

    //switch (move)
    //{
    //case 1:
    //    cudaKernelLS_M1 << <1, u.getTour().size(), 0, stream >> >(routeU, u.getDepot(), u.getTour().size(), u.getCost(), u.getDemand(), u.getServiceTime(),
    //        routeV, v.getDepot(), v.getTour().size(), v.getCost(), v.getDemand(), v.getServiceTime(),
    //        bestChange, bestCost,
    //        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
    //        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
    //        u.getProblem()->getCapacity(), u.getProblem()->getDuration());
    //    break;

    //case 4:
    //    cudaKernelLS_M4 << <1, u.getTour().size(), 0, stream >> >(routeU, u.getDepot(), u.getTour().size(), u.getCost(), u.getDemand(), u.getServiceTime(),
    //        routeV, v.getDepot(), v.getTour().size(), v.getCost(), v.getDemand(), v.getServiceTime(),
    //        bestChange, bestCost,
    //        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
    //        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
    //        u.getProblem()->getCapacity(), u.getProblem()->getDuration());
    //    break;
    //}

    //gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    // https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialWhenSomethingGoesWrong
    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    //---result = cudaPerformeChange(u, v, routeU, routeV, bestChange, move);

    if (result)
        improved = true;

    //if (u.getTour().size() == 0)
    //    break;

    //} while (result == true && !u.getProblem()->getMonitor().isTerminated());

    cudaStreamDestroy(stream);
    cudaFree(routeU);
    cudaFree(routeV);
    cudaFree(bestChange);
    cudaFree(bestCost);

    //cout << "End LS: " << move << " - U: " << u.getDepot() << " - V: " << v.getDepot() << endl;

    return improved;

}

bool cudaLocalSearch(Route& u, Route& v, int move, int streamId) {

    bool result = false;
    bool improved = false;

    //cout << "GPU LS: " << move << " - ID: " << streamId << endl;

    if (move == 7)
        return cudaOperateMoveDepotRouteM7(u, streamId);

    if (move == 8)
        return cudaOperateMoveDepotRouteM8(u, v, streamId);

    if (move == 9)
        return cudaOperateMoveDepotRouteM9(u, v, streamId);

    int sizeU = (int)u.getTour().size();
    int sizeV = (int)v.getTour().size();

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());
    thrust::device_vector<int> d_v(v.getTour().begin(), v.getTour().end());
    thrust::device_vector<int> d_bestChange(sizeU + sizeV);
    thrust::device_vector<float> d_bestCost(sizeU + sizeV);

    //cudaStreamAttachMemAsync(stream, convertToKernel(d_u)._array);
    //cudaStreamAttachMemAsync(stream, convertToKernel(d_v)._array);
    //cudaStreamAttachMemAsync(stream, convertToKernel(d_bestChange)._array);
    //cudaStreamAttachMemAsync(stream, convertToKernel(d_bestCost)._array);
    //cudaStreamSynchronize(stream);

    switch (move)
    {
    case 1:
        cudaKernelLS_M1 << <1, u.getTour().size() >> >(convertRouteToKernel(d_u, u.getDepot(), u.getCost(), u.getDemand(), u.getServiceTime()),
            convertRouteToKernel(d_v, v.getDepot(), v.getCost(), v.getDemand(), v.getServiceTime()),
            convertToKernel(d_bestChange), convertToKernel(d_bestCost),
            u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
            u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
            u.getProblem()->getCapacity(), u.getProblem()->getDuration());
        break;

    case 4:
        cudaKernelLS_M4 << <1, u.getTour().size() >> >(convertRouteToKernel(d_u, u.getDepot(), u.getCost(), u.getDemand(), u.getServiceTime()),
            convertRouteToKernel(d_v, v.getDepot(), v.getCost(), v.getDemand(), v.getServiceTime()),
            convertToKernel(d_bestChange), convertToKernel(d_bestCost),
            u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
            u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
            u.getProblem()->getCapacity(), u.getProblem()->getDuration());
        break;
    }

    //cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    // https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialWhenSomethingGoesWrong
    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    int bi = d_bestChange[0];
    int bj = d_bestChange[1];

    result = cudaPerformeChange(u, v, d_u, d_v, bi, bj, move);

    if (result)
        improved = true;

    //cudaStreamDestroy(stream);

    return improved;

}