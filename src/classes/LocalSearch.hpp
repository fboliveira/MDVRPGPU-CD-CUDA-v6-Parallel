/* 
 * File:   LocalSearch.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 28, 2014, 2:28 PM
 */

#ifndef LOCALSEARCH_HPP
#define	LOCALSEARCH_HPP

#include <list>
#include <algorithm>
#include "MDVRPProblem.hpp"
#include "AlgorithmConfig.hpp"
#include "Route.hpp"

class LocalSearch {

//    MDVRPProblem *problem;
//    AlgorithmConfig *config;
    
public:
//    LocalSearch(MDVRPProblem* problem, AlgorithmConfig* config);
//    LocalSearch(const LocalSearch& orig);
//    virtual ~LocalSearch();
      
//    AlgorithmConfig* getConfig() const;
//    void setConfig(AlgorithmConfig *config);
//
//    MDVRPProblem* getProblem() const;
//    void setProblem(MDVRPProblem *problem);
    
    static bool processMoveDepotRoute(Route& ru, Route& rv, int move, bool equal);
    static bool operateMoves(MDVRPProblem *problem, AlgorithmConfig *config, Route& ru, Route& rv, bool equal);
    
    
private:

    static bool operateMoveDepotRouteFacade(Route& ru, Route& rv, int move, bool equal);
    static bool operateMoveDepotRouteM1(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM2(Route& ru, Route& rv, bool equal, bool operateM3 = false);
    static bool operateMoveDepotRouteM3(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM4(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM5(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM6(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM7(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM8(Route& ru, Route& rv, bool equal);
    static bool operateMoveDepotRouteM9(Route& ru, Route& rv, bool equal);
    
};

#endif	/* LOCALSEARCH_HPP */

