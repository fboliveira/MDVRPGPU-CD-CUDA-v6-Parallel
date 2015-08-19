/* 
 * File:   Pairing.hpp
 * Author: fernando
 *
 * Created on July 22, 2014, 5:29 PM
 */

#ifndef PAIRING_HPP
#define	PAIRING_HPP

#include <vector>
#include <cfloat>

#include "Random.hpp"
#include "AlgorithmConfig.hpp"
#include "MDVRPProblem.hpp"

using namespace std;

class Pairing {
    
    MDVRPProblem* problem;
    AlgorithmConfig* config;

    int depot;
    int individualId;
    
    vector<int> depotRelation;
    float cost;

public:

    Pairing(MDVRPProblem* problem, AlgorithmConfig* config, int depot, int individualId);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig* config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem* problem);

    float getCost() const;
    void setCost(float cost);

    int getDepot() const;
    void setDepot(int depot);

    int getIndividualId() const;
    void setIndividualId(int individualId);
    
    vector<int>& getDepotRelation();
    void setDepotRelation(vector<int> depotRelation);

    void create();
    void pairingRandomly();
    void pairingAllVsBest();

    void print();
    
private:

};

#endif	/* PAIRING_HPP */

