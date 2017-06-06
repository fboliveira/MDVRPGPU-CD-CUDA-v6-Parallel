/* 
 * File:   PathRelinking.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on October 15, 2014, 10:36 AM
 */

#ifndef PATHRELINKING_HPP
#define	PATHRELINKING_HPP

#include <vector>
#include "IndividualsGroup.hpp"
#include "CustomerPosition.hpp"

class PathRelinking {
    
    MDVRPProblem* problem;
    AlgorithmConfig* config;
    
    vector<CustomerPosition> initialPositions;
    vector<CustomerPosition> guidePositions;
    vector<int> difference;

public:
    PathRelinking(MDVRPProblem* problem, AlgorithmConfig* config);
    PathRelinking(const PathRelinking& orig);
    virtual ~PathRelinking();
    
    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem* problem);
    
    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig* config);
    
    /*Methods*/
    void operate(IndividualsGroup& initialSolution, IndividualsGroup& guideSolution);
    
private:

    vector<CustomerPosition>& getInitialPositions();
    void setInitialPositions(vector<CustomerPosition> initialPositions);
    
    vector<CustomerPosition>& getGuidePositions();
    void setGuidePositions(vector<CustomerPosition> guidePositions);

    vector<int>& getDifference();
    void setDifference(vector<int> difference);
    
    /*Methods*/
    void differenceBetweenSolutions();
    
};

#endif	/* PATHRELINKING_HPP */

