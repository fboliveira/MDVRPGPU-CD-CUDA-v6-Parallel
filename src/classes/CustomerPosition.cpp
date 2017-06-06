/* 
 * File:   CustomerPosition.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on October 10, 2014, 6:09 PM
 */

#include "CustomerPosition.hpp"

/*
 * Constructors and Destructor
 */

CustomerPosition::CustomerPosition() {
    this->setCustomer(-1);
    this->setDepot(-1);
    this->setRoute(-1);
}

CustomerPosition::CustomerPosition(const CustomerPosition& other) :
    customer(other.customer), depot(other.depot), route(other.route) {
}

CustomerPosition::~CustomerPosition() {
}

/*
 * Getters and Setters
 */

int CustomerPosition::getCustomer() const {
    return customer;
}

void CustomerPosition::setCustomer(int customer) {
    this->customer = customer;
}

int CustomerPosition::getDepot() const {
    return depot;
}

void CustomerPosition::setDepot(int depot) {
    this->depot = depot;
}

int CustomerPosition::getRoute() const {
    return route;
}

void CustomerPosition::setRoute(int route) {
    this->route = route;
}

/*
 * Public Methods
 */

bool CustomerPosition::isValid() {
    return this->getCustomer() >= 0
            && this->getDepot() >= 0
            && this->getRoute() >= 0;
}


void CustomerPosition::print() {
    cout << "C: " << this->getCustomer() << "\tD: " << this->getDepot() << "\tR: " << this->getRoute() << endl;   
}

bool CustomerPosition::operator==(const CustomerPosition& right) const {
    bool result = false; // Compare right and *this here

    result = right.getCustomer() == this->getCustomer()
            && right.getDepot() == this->getDepot()
            && right.getRoute() == this->getRoute();

    return result;
}

bool CustomerPosition::operator!=(const CustomerPosition& right) const {
    bool result = !(*this == right); // Reuse equals operator
    return result;
}

/*
 * Private Methods
 */
