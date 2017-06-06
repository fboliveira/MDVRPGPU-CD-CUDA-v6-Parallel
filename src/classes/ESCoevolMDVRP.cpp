/* 
 * File:   ESCoevolMDVRP.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 24, 2014, 3:50 PM
 */

#include "ESCoevolMDVRP.hpp"
#include "PathRelinking.hpp"

#include "../cuda/cuda_local_search.h"
//#include "../cuda/thrust/cuda_thrust_test.h"

#include "cuda_profiler_api.h"


/*
 * Constructors and Destructor
 */

ESCoevolMDVRP::ESCoevolMDVRP(MDVRPProblem *problem, AlgorithmConfig *config) :
problem(problem), config(config) {
}

/*
 * Getters and Setters
 */

AlgorithmConfig* ESCoevolMDVRP::getConfig() const {
    return this->config;
}

void ESCoevolMDVRP::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* ESCoevolMDVRP::getProblem() const {
    return this->problem;
}

void ESCoevolMDVRP::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

/*
 * Public Methods
 */

/*
 * ES - Evolution Strategy - Coevolucionary - MDVRP
 */
void ESCoevolMDVRP::run() {

    // # Instanciar variaveis 
    time_t start;
    time(&start);

    //ESCoevolMDVRP::testFunction2(this->getProblem(), this->getConfig());
    //return;
    
    this->getProblem()->getMonitor().setStart(start);

    // Elite group
    EliteGroup *eliteGroup = new EliteGroup(this->getProblem(), this->getConfig());

    // Create Monitor Locks
    this->getProblem()->getMonitor().createLocks(this->getProblem()->getDepots(), this->getConfig()->getNumSubIndDepots());
    
    // Create structure and subpopulation for each depot       
    Community *community = new Community(this->getProblem(), this->getConfig(), eliteGroup);
    community->pairingRandomly();

    // Evaluate Randomly
    community->evaluateSubpops(true);

    // Associate all versus best
    community->pairingAllVsBest();

    // Evaluate All Vs Best
    community->evaluateSubpops(true);

    //ESCoevolMDVRP::testFunction(this->getProblem(), this->getConfig(), community);
    //return;
    
    // Print evolution
    community->printEvolution(true);
    this->getProblem()->getMonitor().updateGeneration();
    //eliteGroup->getBest().printSolution();

    // ##### Start manager ###### ----------------
    //try {
        community->manager();
    //}    catch (exception& e) {
    //    cout << e.what();
    //}
    // ########################## ----------------

    // Print result    
    if (this->getConfig()->isSaveLogRunFile())
        community->writeLogToFile();
    
    // Print final solution
    community->getEliteGroup()->getBest().printSolution();

    cout << "\nCustomers: " << community->getEliteGroup()->getBest().getNumTotalCustomers() << endl;
    cout << "Route Customers: " << community->getEliteGroup()->getBest().getNumTotalCustomersFromRoutes() << endl << endl;
    
    community->getEliteGroup()->printValues();

    //community->printSubpopList();
    
    // Clear memory
    delete community;
    
    // Destroy Monitor Locks
    this->getProblem()->getMonitor().destroyLocks(this->getProblem()->getDepots());
    
}

/*
 * Private Methods
 */


void ESCoevolMDVRP::testFunction1(MDVRPProblem* problem, AlgorithmConfig* config) {
    
    int** d_factor = new int*[problem->getCustomers()];
    
    for(int i = 0; i < problem->getCustomers(); ++i)
        d_factor[i] = new int[problem->getCustomers()];
    
    // granularity threshold value θ = βz (where β is a sparsification factor and z is the average cost of the edges)
    float beta = 1.2f;
    
    // distance factorφij =2cij +δj(∀i ∈ I, j ∈ J)isnotgreaterthanthemaximumduration D.
    
    float threshold = beta * problem->getAvgCustomerDistance();
    
    cout << endl << endl;
    cout << "Avg Customer Distance: " << problem->getAvgCustomerDistance() << endl;
    cout << "Granularity threshold: " << threshold << endl;
    cout << endl << endl;
    
    for(int i = 0; i < problem->getCustomers(); ++i) {
        for(int j = 0; j < problem->getCustomers(); ++j) {
            
            d_factor[i][j] = 0;
            
            if (i != j)
                if (2 * problem->getCustomerDistances().at(i).at(j) <= threshold)
                    d_factor[i][j] = 1;
        }
    }
    
    for(int i = 0; i < problem->getCustomers(); ++i) {
        for(int j = 0; j < problem->getCustomers(); ++j) {
            cout << d_factor[i][j] << " ";
        }
        cout << endl;
    }                        

    for (int i = 0; i < problem->getCustomers(); ++i)
        delete [] d_factor[i];
    delete [] d_factor;

}

void ESCoevolMDVRP::testFunction2(MDVRPProblem* problem, AlgorithmConfig* config) {
     
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

    u.printSolution();
    v.printSolution();
    cout << "Cost U+V: " << u.getCost() + v.getCost() << endl;

    cudaOperateMoveDepotRouteM9(u, v, 0);

    cout << "NEW Cost U+V: " << u.getCost() + v.getCost() << endl;

    getchar();
    
}

void ESCoevolMDVRP::testProfile(MDVRPProblem* problem, AlgorithmConfig* config) {

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

    u.printSolution();
    v.printSolution();
    cout << "Cost U+V: " << u.getCost() + v.getCost() << endl;

    bool result = false;

    do{
        cudaProfilerStart();
        result = cudaLocalSearch(u, v, 1, 0);
        cudaProfilerStop();
        cout << result << endl;
    } while (result == true);

    u.printSolution();
    v.printSolution();
    //u.calculateCost();
    //u.printSolution();


}

void ESCoevolMDVRP::testFunction(MDVRPProblem* problem, AlgorithmConfig* config, Community* community) {

    //IndividualsGroup offsprings = IndividualsGroup(problem, config, 0);
    //CudaSubpop::process(community->getSubpops().at(0), offsprings);
    
}
