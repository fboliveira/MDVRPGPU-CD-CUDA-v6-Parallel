/* 
 * File:   Individual.hpp
 * Author: fernando
 *
 * Created on July 21, 2014, 2:05 PM
 */

#ifndef INDIVIDUAL_HPP
#define	INDIVIDUAL_HPP

#include <vector>
#include <iterator>
#include <list>
#include <cfloat>
#include <algorithm>

#include "MDVRPProblem.hpp"
#include "AlgorithmConfig.hpp"
#include "Route.hpp"
#include "Util.hpp"
#include "Random.hpp"
#include "LocalSearch.hpp"
#include "Lock.hpp"

using namespace std;

class Individual {

    MDVRPProblem *problem;
    AlgorithmConfig *config;

    int depot;
    int id;
    
    bool changed;

    vector<int> gene;
    vector<Route> routes; // Trips with route delimiter - created, at first, using split algorithm.
    
    bool locked = false;
    
    /*Self-adaptive parameters*/
    int numOperations;
    
    float mutationRatePM;
    float mutationRatePLS;
    bool relaxSplit;
    bool restartMoves;
    
public:

    Individual();
    Individual(MDVRPProblem* problem, AlgorithmConfig* config, int depot, int id);
    // Copy constructor    
    Individual(const Individual& other);
    
    vector<int>& getGene();
    vector<int> getGeneConst() const;
    void setGene(vector<int> gene);

    vector<Route>& getRoutes();
    vector<Route> getRoutesConst() const;    
    void setRoutes(vector<Route> routes);

    int getId() const;
    void setId(int id);

    int getDepot() const;
    void setDepot(int depot);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);
    
    bool isChanged() const;
    void setChanged(bool changed);

    bool isLocked() const;
    void setLocked(bool locked);
    
    size_t getNumCustomers();
    size_t getNumCustomersRoute();
    
    int getCustomer(int position);
    
    int getNumOperations() const;
    void setNumOperations(int numOperations);
    
    float getMutationRatePM() const;
    void setMutationRatePM(float mutationRatePM);

    float getMutationRatePLS() const;
    void setMutationRatePLS(float mutationRatePLS);
    
    bool isRelaxSplit() const;
    void setRelaxSplit(bool relaxSplit);    

    bool isRestartMoves() const;
    void setRestartMoves(bool restartMoves);
    
    //std::mutex getMutexLocker() const;
    //void setMutexLocker(std::mutex mutexLocker);

    //std::condition_variable getConditionVariable() const;
    //void setConditionVariable(std::condition_variable conditionVariable);
    
    void setCustomersPosition(vector<CustomerPosition>& position);
    
    /*Methods*/
    
    void add(int customer);
    void add(typedef_vectorIntIterator position, int customer);

    void remove(int customer);

    template<typename Iter>
    void remove(Iter position);

    int find(int customer);

    void create();
    void evaluate(bool split);

    Individual evolve();
    void mutate();
    void localSearch();

    float getTotalCost();
    float getPenaltyVehicles();

    int getNumVehicles();
    
    bool isPenalized();
    
    void split();
    void routesToGenes();

    int autoUpdate(bool update);
    void updateParameters(bool improved);
    
    void removeRepeteadCustomer(int customer);

    typedef_location getMinimalPositionToInsert(int customer);

    Individual copy(bool data);
    void update(Individual& source);
        
    void print();
    void print(bool gene);
    void printSolution(bool insertTotalCost = false);        
    void printVehicles();

private:

    void generateInitialSolutionRandomNearestInsertion();    
    void splitExtract(vector<int>& P);
    void splitExtractWithoutVehicleLimit(vector<int>& P);
    void splitExtractVehicleLimited(vector<int>& P);
    void updateRoutesID();
    
};

#endif	/* INDIVIDUAL_HPP */

