/* 
 * File:   MDVRPProblem.hpp
 * Author: fernando
 *
 * Created on July 21, 2014, 2:10 PM
 */

#ifndef MDVRPPROBLEM_HPP
#define	MDVRPPROBLEM_HPP

#include <vector>
#include <string>
#include <time.h>
#include <cmath>
#include <iostream>

#include "../global.hpp"
#include "Util.hpp"
#include "Monitor.hpp"

using namespace std;

class MDVRPProblem {
private:

    std::string instance;
    std::string instCode;
    float bestKnowSolution;
    int vehicles;
    int customers;
    int depots;
    // Homogeneous fleet
    int capacity;
    float duration;
    vector<typedef_point> customerPoints;
    vector<typedef_point> depotPoints;
    vector<int> demand;

    typedef_vectorMatrix<int> nearestCustomerFromCustomer; // List of nearest customers from customers - CxC
    typedef_vectorMatrix<int> nearestDepotsFromCustomer; // List of nearest depots from customers - CxD
    typedef_vectorMatrix<int> nearestCustomersFromDepot; // List of nearest customers from depot - DxC

    typedef_vectorMatrix<float> customerDistances; //[CLI][CLI];
    typedef_vectorMatrix<float> depotDistances; //[DEP][CLI];

    typedef_vectorMatrix<int> allocation; // In which depots customers are allocated [ c x d ]

    Monitor monitor = Monitor();
    
    double avgCustomerDistance;
    double avgDepotDistance;
    
    typedef_vectorMatrix<int> granularNeighborhood;    

public:

    MDVRPProblem();

    typedef_vectorMatrix<int>& getAllocation();
    float getBestKnowSolution() const;
    void setBestKnowSolution(float bestKnowSolution);
    int getCapacity() const;
    void setCapacity(int capacity);
    typedef_vectorMatrix<float>& getCustomerDistances();
    vector<typedef_point>& getCustomerPoints();
    int getCustomers() const;
    void setCustomers(int customers);
    vector<int>& getDemand();
    
    int getDepots() const;
    void setDepots(int depot);
    
    typedef_vectorMatrix<float>& getDepotDistances();
    vector<typedef_point>& getDepotPoints();

    float getDuration() const;
    void setDuration(float duration);

    std::string getInstCode() const;
    void setInstCode(std::string instCode);

    std::string getInstance() const;
    void setInstance(std::string instance);

    typedef_vectorMatrix<int>& getNearestCustomerFromCustomer();
    typedef_vectorMatrix<int>& getNearestCustomersFromDepot();
    typedef_vectorMatrix<int>& getNearestDepotsFromCustomer();

    int getVehicles() const;
    void setVehicles(int vehicles);

    void setAllocation(typedef_vectorMatrix<int> allocation);
    void setCustomerDistances(typedef_vectorMatrix<float> customerDistances);
    void setCustomerPoints(vector<typedef_point> customerPoints);
    void setDemand(vector<int> demand);
    void setDepotDistances(typedef_vectorMatrix<float> depotDistances);
    void setDepotPoints(vector<typedef_point> depotPoints);
    void setNearestCustomerFromCustomer(typedef_vectorMatrix<int> nearestCustomerFromCustomer);
    void setNearestCustomersFromDepot(typedef_vectorMatrix<int> nearestCustomersFromDepot);
    void setNearestDepotsFromCustomer(typedef_vectorMatrix<int> nearestDepotsFromCustomer);

    bool processInstanceFiles(char *dataFile, char *solutionFile, char* instCode);

    void getDepotGroup(int depot, vector<int>& customers);
    void getNearestCustomerFromCustomerOnDepot(int customer, int depot, vector<int>& customers);

    Monitor& getMonitor();

    void setAvgDepotDistance(double avgDepotDistance);
    double getAvgDepotDistance() const;

    void setAvgCustomerDistance(double avgCustomerDistance);
    double getAvgCustomerDistance() const;
    
    typedef_vectorMatrix<int>& getGranularNeighborhood();
    void setGranularNeighborhood(typedef_vectorMatrix<int> granularNeighborhood);
    
    void print();
    void printAllocation();
    void printAllocationDependecy();

private:

    void allocateMemory();
    void calculateMatrixDistance();
    void setNearestCustomersFromCustomer();
    void setNearestDepotsFromCustomer();
    void setNearestCustomersFromDepot();

    void defineIntialCustomersAllocation();
    void setCustomerOnDepot(int customer);
    
    void operateGranularNeighborhood();

};

#endif	/* MDVRPPROBLEM_HPP */

