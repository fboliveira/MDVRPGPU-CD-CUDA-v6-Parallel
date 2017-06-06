/* 
 * File:   Pairing.cpp
 * Author: fernando
 * 
 * Created on July 22, 2014, 5:29 PM
 */

#include "Pairing.hpp"

/*
 * Constructors
 */


Pairing::Pairing(MDVRPProblem* problem, AlgorithmConfig* config, int depot, int individualId) :
    problem(problem), config(config),  depot(depot), individualId(individualId) {

    this->setCost(FLT_MAX);
    
}

/*
 * Getters and Setters
 */

float Pairing::getCost() const {
    return this->cost;
}

void Pairing::setCost(float cost) {
    this->cost = cost;
}

int Pairing::getDepot() const {
    return this->depot;
}

void Pairing::setDepot(int depot) {
    this->depot = depot;
}

int Pairing::getIndividualId() const {
    return this->individualId;
}

void Pairing::setIndividualId(int individualId) {
    this->individualId = individualId;
}

vector<int>& Pairing::getDepotRelation() {
    return this->depotRelation;
}

void Pairing::setDepotRelation(vector<int> depotRelation) {
    this->depotRelation = depotRelation;
}

AlgorithmConfig* Pairing::getConfig() const {
    return this->config;
}

void Pairing::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* Pairing::getProblem() const {
    return this->problem;
}

void Pairing::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

/*
 * Methods
 */

void Pairing::create() {
    
    vector<int> depotRelation = vector<int>(this->getProblem()->getDepots(), -1);
    this->setDepotRelation(depotRelation);
    
}

void Pairing::pairingRandomly() {

    this->create();
    for(auto iter = this->getDepotRelation().begin(); iter != this->getDepotRelation().end(); ++iter) {
        *iter = Random::randIntBetween(0, this->getConfig()->getNumSubIndDepots() - 1);
    }

}

void Pairing::pairingAllVsBest() {

    this->create();
    this->getDepotRelation().at( this->getDepot() ) = this->getIndividualId();

}

void Pairing::print() {

    cout << "Pairing::print(): Dep: " << this->getDepot() << "  Id: ";
    cout << this->getIndividualId() << " Cost: " << this->getCost() << " => ";
    Util::print(this->getDepotRelation());
    //cout << endl;
    
}
