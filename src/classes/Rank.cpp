/* 
 * File:   Rank.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on November 6, 2014, 4:41 PM
 */

#include "Rank.hpp"
#include "Util.hpp"

/*
 * Constructors and Destructor
 */

Rank::Rank() {
}

Rank::Rank(int source, int id, float cost) :
    source(source), id(id), cost(cost) {
}

Rank::Rank(const Rank& other) :
    source(other.source), id(other.id), cost(other.cost) {
}


Rank::~Rank() {
}
/*
 * Getters and Setters
 */

int Rank::getSource() const {
    return source;
}

void Rank::setSource(int source) {
    this->source = source;
}

int Rank::getId() const {
    return id;
}

void Rank::setId(int id) {
    this->id = id;
}

float Rank::getCost() const {
    return cost;
}

void Rank::setCost(float cost) {
    this->cost = cost;
}
/*
 * Public Methods
 */

void Rank::print() {
    cout << "S: " << this->getSource() << " - id: " << this->getId() << " - Cost: " << this->getCost() << endl;
}

bool Rank::compare(Rank i, Rank j) {
    return Util::isBetterSolution(i.getCost(), j.getCost());
}

/*
 * Private Methods
 */
