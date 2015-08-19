/* 
 * File:   EliteGroup.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 24, 2014, 11:40 AM
 */

#ifndef ELITEGROUP_HPP
#define	ELITEGROUP_HPP

#include <vector>
#include <algorithm>

#include "Util.hpp"
#include "IndividualsGroup.hpp"

using namespace std;

class EliteGroup {
    
    vector<IndividualsGroup> eliteGroup;
    vector<IndividualsGroup> pool;

    IndividualsGroup best;
    
    MDVRPProblem *problem;
    AlgorithmConfig *config;
    
    bool changed;
    bool locked = false;
    
public:
    
    EliteGroup();
    EliteGroup(MDVRPProblem *problem, AlgorithmConfig *config);

    vector<IndividualsGroup>& getEliteGroup();
    void setEliteGroup(vector<IndividualsGroup> eliteGroup);

    IndividualsGroup& getBest();
    void setBest(IndividualsGroup best);   
    
    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);
    
    vector<IndividualsGroup>& getPool();
    void setPool(vector<IndividualsGroup> pool);
   
    bool isChanged() const;
    void setChanged(bool changed);
    
    bool isLocked() const;
    void setLocked(bool locked);
    
    void update(IndividualsGroup& individuals);    
    void updatePool(IndividualsGroup& individuals);    
    
    void localSearch();
    
    void manager();
    void printValues();

    void managerOLD();
    void localSearchOLD();
    
private:
    
};

#endif	/* ELITEGROUP_HPP */

