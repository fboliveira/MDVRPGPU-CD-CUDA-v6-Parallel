/* 
 * File:   ESCoevolMDVRP.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 24, 2014, 3:50 PM
 */

#ifndef ESCOEVOLMDVRP_HPP
#define	ESCOEVOLMDVRP_HPP

#include "MDVRPProblem.hpp"
#include "AlgorithmConfig.hpp"
#include "Community.hpp"
#include "Util.hpp"

class ESCoevolMDVRP {
    
    MDVRPProblem *problem;
    AlgorithmConfig *config;
    
public:

    ESCoevolMDVRP(MDVRPProblem *problem, AlgorithmConfig *config);
    
    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);
    
    void run();

private:

    static void testFunction1(MDVRPProblem* problem, AlgorithmConfig* config);
    static void testFunction2(MDVRPProblem* problem, AlgorithmConfig* config);
    static void testProfile(MDVRPProblem* problem, AlgorithmConfig* config);
    static void testFunction(MDVRPProblem* problem, AlgorithmConfig* config, Community *community);
    
};

#endif	/* ESCOEVOLMDVRP_HPP */

