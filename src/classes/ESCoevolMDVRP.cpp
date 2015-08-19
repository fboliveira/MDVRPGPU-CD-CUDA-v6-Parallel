/* 
 * File:   ESCoevolMDVRP.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 24, 2014, 3:50 PM
 */

#include "ESCoevolMDVRP.hpp"
#include "PathRelinking.hpp"

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

    ESCoevolMDVRP::testFunction2(this->getProblem(), this->getConfig());
    return;
    
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
    community->printEvolution();
    this->getProblem()->getMonitor().updateGeneration();
    //eliteGroup->getBest().printSolution();

    // ##### Start manager ###### ----------------
    try {
        community->manager();
    }    catch (exception& e) {
        cout << e.what();
    }
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
    float beta = 1.2;
    
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
    
//    int** sol = new int*[problem->getCustomers()];
//    
//    for(int i = 0; i < problem->getCustomers(); ++i)
//        sol[i] = new int[problem->getCustomers()];
//        
//    for (int i = 0; i < problem->getCustomers(); ++i)
//        delete [] sol[i];
//    delete [] sol;
  
    int s[] = {72,155,44,38,27,110,230,41,78,90,243,203};
    int N = 12;
    Route r = Route(problem, config, 4, 0);
    
    for(int i = 0; i < N; ++i)
        r.addAtBack(s[i]);
    
    r.printSolution();
    
    
}

void ESCoevolMDVRP::testFunction(MDVRPProblem* problem, AlgorithmConfig* config, Community* community) {

    PathRelinking pathRelinking = PathRelinking(problem, config);
    
    community->getEliteGroup()->getEliteGroup().at(2).printSolution();
    
    pathRelinking.operate(community->getEliteGroup()->getEliteGroup().at(2), community->getEliteGroup()->getBest());

    community->getEliteGroup()->getEliteGroup().at(2).printSolution();
    
}
