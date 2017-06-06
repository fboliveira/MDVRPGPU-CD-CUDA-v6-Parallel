/* 
 * File:   IndividualsGroup.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 23, 2014, 5:03 PM
 */

#ifndef INDIVIDUALSGROUP_HPP
#define	INDIVIDUALSGROUP_HPP

#include <vector>
#include <iterator>
#include <list>
#include <cfloat>
#include <algorithm>

#include "Individual.hpp"
#include "Random.hpp"
#include "Util.hpp"
#include "CustomerPosition.hpp"
#include "Rank.hpp"

using namespace std;

class IndividualsGroup {

    MDVRPProblem *problem; 
    AlgorithmConfig *config;
    
    int depot;
    int id;

    vector<Individual> individuals;
    
    bool locked = false;
    bool lsProcessed = false;
    
    /*Self-adaptive*/
    bool forceSequential = false;
    
    vector<Rank> ranks;
        
public:
    
    IndividualsGroup();
    IndividualsGroup(MDVRPProblem* problem, AlgorithmConfig* config, int depot);
    IndividualsGroup(const IndividualsGroup& orig);

    int getDepot() const;
    void setDepot(int depot);
    
    int getId() const;
    void setId(int id);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);

    vector<Individual>& getIndividuals();
    void setIndividuals(vector<Individual> individuals);       
    vector<Individual> getIndividualsConst() const;
    
    bool isLocked() const;
    void setLocked(bool locked);
    
    bool isLSProcessed() const;
    void setLSProcessed(bool lsProcessed);
    
    bool isForceSequential() const;
    void setForceSequential(bool forceSequential);
    
    vector<Rank>& getRanks();
    void setRanks(vector<Rank> ranks);
    
    vector<CustomerPosition> getCustomersPosition();    
    
    void clear();
    void add(Individual individual);
    size_t size();
    
    size_t getNumTotalCustomers();
    size_t getNumTotalCustomersFromRoutes();
    
    void evaluate(bool removeConflicts, bool split);
    void localSearch(bool fullImprovement = false, bool runOnGPU = false);
    
    float getTotalCost();
    float getIncompleteSolutionPenalty();
    
    bool isChanged();
    bool isPenalized();

    //void merge(IndividualsGroup& source);
    void rank(int source);
    void shrink(IndividualsGroup& source);
    
    void print();
    void print(bool gene);
    void printSolution();
    void printList();
    void printRanks();
    
private:

    void removeConflictedCustomersFromDepots();
    static bool compareIndividuals(Individual i, Individual j);
    
};

#endif	/* INDIVIDUALSGROUP_HPP */