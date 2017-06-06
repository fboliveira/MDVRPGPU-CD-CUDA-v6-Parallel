/* 
 * File:   Subpopulation.hpp
 * Author: fernando
 *
 * Created on July 22, 2014, 4:38 PM
 */

#ifndef SUBPOPULATION_HPP
#define	SUBPOPULATION_HPP

#include <thread>         // std::thread
#include <vector>         // std::vector

using namespace std;

#include <atomic>         // std::atomic

#include "../global.hpp"
#include "Individual.hpp"
#include "IndividualsGroup.hpp"
#include "Pairing.hpp"
#include "EliteGroup.hpp"
#include "Lock.hpp"

#include "../cuda/cuda_types.h"
#include "../cuda/cuda_subpop.h"

class Subpopulation {

    MDVRPProblem *problem;
    AlgorithmConfig *config;
    EliteGroup* eliteGroup;
    
    int depot;
    Individual best = Individual(this->getProblem(), this->getConfig(), this->getDepot(), -1);
    
    IndividualsGroup individuals;
    vector<Pairing> pairing;
    
    bool locked = false;

    StrIndividual *mngPop;
    StrIndividual *mngPopRes;
    
    //Lock* locks;
        
public:
   
    Subpopulation();
    Subpopulation(MDVRPProblem *problem, AlgorithmConfig *config, EliteGroup* eliteGroup, int depot);
    Subpopulation(const Subpopulation& other);
    virtual ~Subpopulation();
           
    Individual& getBest();
    void setBest(Individual best);

    int getDepot() const;
    void setDepot(int depot);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);
    
    EliteGroup* getEliteGroup();
    void setEliteGroup(EliteGroup* eliteGroup);

    IndividualsGroup& getIndividualsGroup();
    void setIndividualsGroup(IndividualsGroup individuals);

    vector<Pairing>& getPairing();
    void setPairing(vector<Pairing> solutions);
    
    bool isLocked() const;
    void setLocked(bool locked);
    
    void createIndividuals();
    
    void createPairingStructure();
    void pairingRandomly();
    void pairingAllVsBest();
    void printPairing();
    
    void evolve();
    void evolveCPU();
    void evolveGPU();

    //Lock& getLock(int id);
    
private:

    //Lock* getLocks();
    //void createLockers();
    void copyTOManaged();
    void copyFROMManaged(IndividualsGroup& offsprings);
    void localSearchOffsprings(IndividualsGroup& offsprings);
    void deallocateManaged();

};

#endif	/* SUBPOPULATION_HPP */

