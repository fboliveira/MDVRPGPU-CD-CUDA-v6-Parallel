/* 
 * File:   CustomerPosition.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on October 10, 2014, 6:09 PM
 */

#ifndef CUSTOMERPOSITION_HPP
#define	CUSTOMERPOSITION_HPP

#include <iostream>

using namespace std;

class CustomerPosition {
    
    int customer;
    int depot;
    int route;
    
public:
    
    CustomerPosition();   
    CustomerPosition(const CustomerPosition& other);
    virtual ~CustomerPosition();
    
    int getCustomer() const;
    void setCustomer(int customer);

    int getDepot() const;
    void setDepot(int depot);

    int getRoute() const;
    void setRoute(int route);
    
    bool isValid();
    void print();
    
    bool operator==(const CustomerPosition& right) const;
    bool operator!=(const CustomerPosition& right) const;    
    
private:

};

#endif	/* CUSTOMERPOSITION_HPP */

