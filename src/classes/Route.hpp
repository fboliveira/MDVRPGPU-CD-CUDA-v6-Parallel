/* 
 * File:   Route.hpp
 * Author: fernando
 *
 * Created on July 21, 2014, 10:09 PM
 */

#ifndef ROUTE_HPP
#define	ROUTE_HPP

#include <list>
#include <iterator>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <cfloat>

#include "MDVRPProblem.hpp"
#include "AlgorithmConfig.hpp"
#include "CustomerPosition.hpp"

using namespace std;

class Route {
    
    list<int> tour;

    float cost;
    
    float penaltyDuration;
    float penaltyDemand;

    int id;
    int depot;
    int demand;
    int serviceTime;

    bool relaxDuration;
    
    MDVRPProblem *problem; 
    AlgorithmConfig *config;      

public:

    Route(MDVRPProblem *problem, AlgorithmConfig *config, int depot, int routeID);
    // Copy constructor
    //Route(const Route& route);
    
    Route(const Route& other);
    
    float getCost();
    void setCost(float cost);
    
    int getServiceTime() const;
    void setServiceTime(int serviceTime);
    
    bool isRelaxDuration() const;
    void setRelaxDuration(bool relaxDuration);
    
    float getPenaltyDuration() const;
    void setPenaltyDuration(float penalty);

    int getDemand() const;
    void setDemand(int demand);

    float getPenaltyDemand() const;
    void setPenaltyDemand(float penaltyDemand);
    
    int getId() const;
    void setId(int id);

    int getDepot() const;
    void setDepot(int depot);

    list<int> getTourConst() const;
    list<int>& getTour();
    void setTour(list<int> tour);
    
    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);
    
    void setCustomersPosition(vector<CustomerPosition>& position);
    
    float getTotalCost();
    void updatePenalty();
    
    void startValues();
    
    typedef_listIntIterator addAtFront(int customer);    
    typedef_listIntIterator addAtBack(int customer);    

    //template<typename Iter>
    //void addAfterPrevious(Iter previous, int customer);
    typedef_listIntIterator addAfterPrevious(typedef_listIntIterator previous, int customer);
    typedef_listIntIterator addAfterPrevious(int previousCustomer, int customer);
    
    void insertBestPosition(int customer);
    
    typedef_listIntIterator find(int customer);

    //    template<typename Iter>
//    Iter find(int customer);
    
    //template<typename Iter>
    //void remove(Iter position);
    void remove(typedef_listIntIterator position);
    void remove(int customer);
      
    void calculateCost();        
    float calculateCost(typedef_listIntIterator start, typedef_listIntIterator end, int& demand);
    
    void changeCustomer(typedef_listIntIterator position, int newCustomer);
    void swap(typedef_listIntIterator source, typedef_listIntIterator dest);
    void reverse(typedef_listIntIterator begin, typedef_listIntIterator end);
    
    bool isPenalized();
    
    float getDuration() const;
    int getCapacity() const;
    
    void routeToVector(int* route);
    void vectorToRoute(int* route, int size);
    void vectorToRoute(int* route, int first, int last);

    void print();    
    void printSolution();
    
    bool operator==(const Route& right) const;
    
private:
        
};

//template<> void Route::remove<typedef_listIntIterator>(typedef_listIntIterator);
//template<> void Route::addAfterPrevious<typedef_listIntIterator>(typedef_listIntIterator, int);

#endif	/* ROUTE_HPP */

