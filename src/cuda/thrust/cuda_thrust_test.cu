#include "cuda_thrust_test.h"

#define NN 30
#define SELECT 3

typedef thrust::tuple<int, int>            tpl2int;
typedef thrust::device_vector<int>::iterator intiter;
typedef thrust::counting_iterator<int> countiter;
typedef thrust::tuple<intiter, countiter>  tpl2intiter;
typedef thrust::zip_iterator<tpl2intiter>  idxzip;

struct select_unary_op : public thrust::unary_function<tpl2int, int>
{
    __host__ __device__
        int operator()(const tpl2int& x) const
    {
        printf("i = %d \tx = %d\n", x.get<1>(), x.get<0>());
        if ((x.get<1>() % SELECT) == 0)
            return x.get<0>();
        else return -1;
    }
};

struct cuda_printf {

    char name;
    cuda_printf(char _name) : name(_name) {};

    __device__
    void operator()(int &x) {
        printf("c = %c - x = %d\n", name, x);
    }

};

struct perform_distance {

    matrix_special<float> mngCustomerDistances;
    matrix_special<float> mngDepotDistances;

    perform_distance(matrix_special<float> _custDist, matrix_special<float> _depDist) :
        mngCustomerDistances(_custDist), mngDepotDistances(_depDist) {};

    __device__
        void operator() (int &x) {
        printf("x = %d - dep %.2f\n", x, mngDepotDistances.data[get_pos(mngDepotDistances.lines, mngDepotDistances.columns, 0, x - 1)]);
    }

};

struct perform_route : public thrust::unary_function<thrust::tuple<int, int>, int> {

    matrix_special<float> mngCustomerDistances;
    matrix_special<float> mngDepotDistances;
    thrust::device_vector<int> rv;

    perform_route(matrix_special<float> _custDist, matrix_special<float> _depDist, thrust::device_vector<int> _rv) :
        mngCustomerDistances(_custDist), mngDepotDistances(_depDist), rv(_rv) {};

    __device__
    int operator() (const thrust::tuple<int, int> &x) const {
        printf("i = %d \tx = %d - dep %.2f\n", x.get<1>(), x.get<0>(), 
            mngDepotDistances.data[get_pos(mngDepotDistances.lines, mngDepotDistances.columns, 0, x.get<0>() - 1)]);
        return x.get<1>();
    }

    //__device__
    //int operator() (const thrust::tuple<int, int> &x) const {
    //    printf("i = %d \tx = %d\n", x.get<1>(), x.get<0>());
    //    return x.get<0>();
    //}
};

__device__ float cuda_calculate_route_cost_orig(int *route, int size, int depot, int *demand, int *service,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService) {

    float cost = 0.0f;
    *demand = 0;
    *service = 0;

    // From depot to first
    cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines, 
        mngDepotDistances.columns, depot, route[0] - 1)];
    // From depot to last
    cost += mngDepotDistances.data[get_pos(mngDepotDistances.lines, 
        mngDepotDistances.columns, depot, route[size - 1] - 1)];

    int customer = 0;

    for (int i = 0; i < size; ++i) {

        customer = route[i] - 1;
        *demand += mngDemand[customer];
        *service += mngService[customer];

        if (i + 1 < size) {
            int j = route[i + 1] - 1;

            cost += mngCustomerDistances.data[get_pos(mngCustomerDistances.lines, mngCustomerDistances.columns,
                customer, j)];

        }

    }

    return cost;

}

__host__ __device__ void cuda_2opt_swap_orig(int *route, int size, int *newRoute, int i, int k) {

    int p = 0;

    // https://en.wikipedia.org/wiki/2-opt

    printf("i = %d - k = %d => ", i, k);

    // 1. take route[1] to route[i-1] and add them in order to new_route
    for (int c = 0; c <= i - 1; ++c) {
        newRoute[p] = route[c];
        printf("%d\t", newRoute[p]);
        p++;
    }

    // 2. take route[i] to route[k] and add them in reverse order to new_route
    for (int c = k; c >= i; --c) {
        newRoute[p] = route[c];
        printf("%d\t", newRoute[p]);
        p++;
    }

    // 3. take route[k+1] to end and add them in order to new_route
    for (int c = k + 1; c < size; ++c) {
        newRoute[p] = route[c];
        printf("%d\t", newRoute[p]);
        p++;
    }

    printf("\n", newRoute[p]);

}

__global__ void cudaKernelLS_M7_orig(KernelArray<int> u, int depot, float routeCost, int demand, int service,
    matrix_special<float> mngCustomerDistances, matrix_special<float> mngDepotDistances,
    int* mngDemand, int* mngService, int capacity, float duration,
    KernelArray<int> bestI, KernelArray<int> bestK, KernelArray<float> bestCost) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int bi = -1;
    int bk = -1;

    float bcost = FLT_MAX;
    float cost;
   
    bestCost._array[idx] = FLT_MAX;

    if (idx < u._size - 1) {

        int *newRoute = new int[u._size];

        for (int k = idx + 1; k < u._size; ++k) {

            cuda_2opt_swap_orig(u._array, u._size, newRoute, idx, k);
            cost = cuda_calculate_route_cost_orig(newRoute, u._size, depot, &demand, &service,
                mngCustomerDistances, mngDepotDistances, mngDemand, mngService);

            // Check service duration
            if (cost + service <= duration)
                if (cost < bcost) {
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

    __syncthreads();

    if (idx == 0) {
        for (int i = 0; i < u._size; ++i)
            printf("i = %d\tbi = %d\tbk = %d\tbc = %f\n", i, bestI._array[i], bestK._array[i], bestCost._array[i]);
    }

    //// Find best change
    //if (idx == 0) {

    //    float value = thrust::reduce(bestCost._array, bestCost._array + u._size, FLT_MAX,thrust::minimum<float>());
    //    //int pos = thrust::find(bestCost, bestCost + u._size, *value);

    //}

}

void cudaThrustLS_M7(MDVRPProblem* problem, AlgorithmConfig* config) {

    int s[] = { 46, 43, 39, 44, 35, 9, 42 };
    int N = 7;
    Route u = Route(problem, config, 0, 0);

    for (int i = 0; i < N; ++i)
        u.addAtBack(s[i]);

    u.printSolution();

    int size = (int)u.getTour().size();

    thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());

    thrust::device_vector<int> bestI(size);
    thrust::device_vector<int> bestK(size);
    thrust::device_vector<float> bestCost(size);

    cudaKernelLS_M7_orig << <1, 1 >> >(convertToKernel(d_u), 0, u.getCost(), u.getDemand(), u.getServiceTime(),
        u.getProblem()->getMngCustomerDistances().getMatrixData(), u.getProblem()->getMngDepotDistances().getMatrixData(),
        u.getProblem()->getMngDemand().getData(), u.getProblem()->getMngServiceTime().getData(),
        u.getProblem()->getCapacity(), u.getProblem()->getDuration(),
        convertToKernel(bestI), convertToKernel(bestK), convertToKernel(bestCost));
    cudaDeviceSynchronize();

    // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
    thrust::device_vector<float>::iterator iter =
        thrust::min_element(thrust::device, bestCost.begin(), bestCost.end());

    unsigned int position = iter - bestCost.begin();
    int bi = bestI[position];
    int bk = bestK[position];

    // There are two different positions
    if (bi >= 0 && bk > bi) {
        int *newRote = new int[u.getTour().size()];
        thrust::host_vector<int> h_u = d_u;
        cuda_2opt_swap_orig(thrust::raw_pointer_cast(&h_u[0]), size, newRote, bi, bk);

        Util::print(newRote, size);

        u.vectorToRoute(newRote, size);
        u.printSolution();
        delete[] newRote;
    }

}

void cudaThrustTest(MDVRPProblem* problem, AlgorithmConfig* config) {

    cudaThrustLS_M7(problem, config);
    return;

    int s[] = { 35, 9, 42, 46, 43, 39, 44 };
    int N = 7;
    Route u = Route(problem, config, 0, 0);

    for (int i = 0; i < N; ++i)
        u.addAtBack(s[i]);

    int t[] = { 32, 31, 36, 41, 7, 37 };
    int M = 6;

    Route v = Route(problem, config, 1, 0);

    for (int i = 0; i < M; ++i)
        v.addAtBack(t[i]);

    //thrust::device_vector<int> d_u(u.getTour().begin(), u.getTour().end());
    //thrust::device_vector<int> d_v(v.getTour().begin(), v.getTour().end());

    //thrust::for_each(thrust::device, d_u.begin(), d_u.end(), cuda_printf('X'));
    //thrust::for_each(thrust::device, d_v.begin(), d_v.end(), cuda_printf('Y'));

    //thrust::for_each(thrust::device, d_u.begin(), d_u.end(),
    //    perform_distance(problem->getMngCustomerDistances().getMatrixData(),
    //    problem->getMngDepotDistances().getMatrixData()));
    //thrust::for_each(thrust::device, d_v.begin(), d_v.end(),
    //    perform_distance(problem->getMngCustomerDistances().getMatrixData(),
    //    problem->getMngDepotDistances().getMatrixData()));

    // http://stackoverflow.com/questions/30231338/in-cuda-thrust-how-can-i-access-a-vector-elements-neighbor-during-a-for-each
    // http://stackoverflow.com/questions/17484835/get-index-of-vector-inside-cuda-thrusttransform-operator-function

    /*
    thrust::device_vector<int> result(u.getTour().size());
    thrust::device_vector<int> index(u.getTour().size());
    thrust::sequence(thrust::device, index.begin(), index.end());

    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst + u.getTour().size();

    thrust::zip_iterator<thrust::tuple<int, int>> first =
    thrust::make_zip_iterator(thrust::make_tuple(index.begin(), idxfirst));

    thrust::zip_iterator<thrust::tuple<int, int>> last =
    thrust::make_zip_iterator(thrust::make_tuple(d_u.end(), idxlast));

    thrust::for_each(thrust::device, first, last,
    perform_route(problem->getMngCustomerDistances().getMatrixData(),
    problem->getMngDepotDistances().getMatrixData()));

    thrust::transform(thrust::device, first, last, result.begin(),
    perform_route(problem->getMngCustomerDistances().getMatrixData(),
    problem->getMngDepotDistances().getMatrixData()));
    */

    thrust::host_vector<int> A(N);
    thrust::host_vector<int> result(N);
    thrust::sequence(A.begin(), A.end());
    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst + N;

    thrust::device_vector<int> B(u.getTour().begin(), u.getTour().end());
    thrust::device_vector<int> C(u.getTour().size());
    thrust::sequence(C.begin(), C.end());

    cout << "B[0] = " << B[0] << endl;

    idxzip first = thrust::make_zip_iterator(thrust::make_tuple(B.begin(), thrust::make_counting_iterator<int>(0)));
    idxzip  last = thrust::make_zip_iterator(thrust::make_tuple(B.end(), thrust::make_counting_iterator<int>(u.getTour().size())));
    //select_unary_op my_unary_op;

    //thrust::transform(first, last, result.begin(), perform_route(problem->getMngCustomerDistances().getMatrixData(),
    //    problem->getMngDepotDistances().getMatrixData()));

    //thrust::transform(first, last, result.begin(), my_unary_op);
    //thrust::for_each(thrust::device, first, last, my_unary_op);

    //thrust::for_each(first, last, perform_route(problem->getMngCustomerDistances().getMatrixData(),
    //    problem->getMngDepotDistances().getMatrixData(), d_v));

    //std::cout << "Results :" << std::endl;
    //thrust::copy(result.begin(), result.end(), std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //cudaThrustTest(problem, config);

    //u.printSolution();
    //v.printSolution();
    //cout << "Cost U+V: " << u.getCost() + v.getCost() << endl;
}