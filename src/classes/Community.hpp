/* 
 * File:   Community.hpp
 * Author: fernando
 *
 * Created on July 21, 2014, 9:57 PM
 */

#ifndef COMMUNITY_HPP
#define	COMMUNITY_HPP

#include <vector>
#include <cstdio>
#include <stdexcept>
#include <forward_list>
#include <string>
#include <thread>         // std::thread
#include <chrono>         // std::chrono::seconds

#include "Subpopulation.hpp"
#include "Pairing.hpp"
#include "EliteGroup.hpp"
#include "Random.hpp"
#include "Util.hpp"

using namespace std;

class Community {
    
    MDVRPProblem *problem;
    AlgorithmConfig *config;

    EliteGroup *eliteGroup;

    vector<Subpopulation> subpops;
    vector<Subpopulation> subpopsPool;
    forward_list<typedef_evolution> evolution;
        
public:

    Community(MDVRPProblem* problem, AlgorithmConfig* config, EliteGroup *eliteGroup);
    ~Community();

    vector<Subpopulation>& getSubpops();
    void setSubpops(vector<Subpopulation> subpops);

    vector<Subpopulation> getSubpopsConst() const;
    
    vector<Subpopulation>& getSubpopsPool();
    void setSubpopsPool(vector<Subpopulation> subpopsPool);
        
    EliteGroup* getEliteGroup();
    void setEliteGroup(EliteGroup *eliteGroup);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);

    forward_list<typedef_evolution>& getEvolution();
    void setEvolution(forward_list<typedef_evolution> evolution);
    
    void createSubpopulations();

    void pairingRandomly();
    void pairingAllVsBest();
    void printPairing();

    void evaluateSubpops(bool firstEvaluation);
    void printEvolution(bool initialValue = false);

    bool isStopCriteriaMet();
    void writeLogToFile();

    void evolve();
    void evaluate();
    void updateBestIndividuals();
    void manager();
    
    void printSubpopList();
    
private:

    void checkEvolution();
    void evaluateSubpopsAndUpdateEG(bool firstEvaluation);
    void evaluateSubpopsParallelOLD();
    void evaluateSubpopsParallel();
    void evaluateSubpop(int subpopID);
        
};

#endif	/* COMMUNITY_HPP */

