/* 
 * File:   PathRelinking.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on October 15, 2014, 10:36 AM
 */

#include "PathRelinking.hpp"

/*
 * Constructors and Destructor
 */

PathRelinking::PathRelinking(MDVRPProblem* problem, AlgorithmConfig* config) :
problem(problem), config(config) {
}

PathRelinking::PathRelinking(const PathRelinking& orig) {
}

PathRelinking::~PathRelinking() {
}

/*
 * Getters and Setters
 */

MDVRPProblem* PathRelinking::getProblem() const {
    return problem;
}

void PathRelinking::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

AlgorithmConfig* PathRelinking::getConfig() const {
    return config;
}

void PathRelinking::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

/*
 * Public Methods
 */

void PathRelinking::operate(IndividualsGroup& initialSolution, IndividualsGroup& guideSolution) {

    // Setup
    this->setInitialPositions(initialSolution.getCustomersPosition());
    this->setGuidePositions(guideSolution.getCustomersPosition());

    this->differenceBetweenSolutions();

    if (this->getDifference().size() == 0)
        return;

    //Util::print(this->getDifference());

    size_t iteration = 0;
    vector<int> c;
    Random::randPermutation(this->getDifference().size(), 0, c);
    //Util::print(c);

    IndividualsGroup newSolution;

    while (iteration < this->getDifference().size() && !this->getProblem()->getMonitor().isTerminated()) {

        int customer = this->getDifference().at(c.at(iteration));

        CustomerPosition pi = this->getInitialPositions().at(customer);
        CustomerPosition pg = this->getGuidePositions().at(customer);

        if (!pi.isValid()) {
            //cout << "PI invalid!\n";
            //continue;
			break;
        }

        if (!pg.isValid()) {
            //cout << "PG invalid!\n";
            //continue;
			break;
        }

        int route = -10;

        if (pg.getRoute() >= initialSolution.getIndividuals().at(pg.getDepot()).getRoutes().size())
            route = initialSolution.getIndividuals().at(pg.getDepot()).getRoutes().size() - 1;
        else
            route = pg.getRoute();

        try {

            if (initialSolution.getIndividuals().at(pg.getDepot()).getRoutes().at(route).getDemand()
                    + this->getProblem()->getDemand().at(customer) <= this->getProblem()->getCapacity()) {

                newSolution = initialSolution;

                // Remove from Initial Solution
                newSolution.getIndividuals().at(pi.getDepot()).getRoutes().at(pi.getRoute()).remove(pi.getCustomer());

                // Best insertion
                newSolution.getIndividuals().at(pg.getDepot()).getRoutes().at(route).insertBestPosition(pg.getCustomer());                
                //newSolution.getIndividuals().at(pg.getDepot()).getRoutes().at(route).addAtFront(pg.getCustomer());
                LocalSearch::operateMoves(this->getProblem(), this->getConfig(), 
                        newSolution.getIndividuals().at(pg.getDepot()).getRoutes().at(route),
                        newSolution.getIndividuals().at(pg.getDepot()).getRoutes().at(route), true);

                if (Util::isBetterSolution(newSolution.getTotalCost(), initialSolution.getTotalCost())) {
                    // Move
                    newSolution.getIndividuals().at(pg.getDepot()).routesToGenes();
                    initialSolution = newSolution;
                }
            }

        } catch (exception& e) {

            pi.print();
            pg.print();

            cout << "It: " << iteration << "\tRoute: " << route << "\t"
                    << "S: " << initialSolution.getIndividuals().at(pi.getDepot()).getRoutes().size() << "\t"
                    << e.what() << endl << endl;
        }

        iteration++;

    }

}

/*
 * Private Methods
 */

vector<CustomerPosition>& PathRelinking::getInitialPositions() {
    return initialPositions;
}

void PathRelinking::setInitialPositions(vector<CustomerPosition> initialPositions) {
    this->initialPositions = initialPositions;
}

vector<CustomerPosition>& PathRelinking::getGuidePositions() {
    return guidePositions;
}

void PathRelinking::setGuidePositions(vector<CustomerPosition> guidePositions) {
    this->guidePositions = guidePositions;
}

vector<int>& PathRelinking::getDifference() {
    return difference;
}

void PathRelinking::setDifference(vector<int> difference) {
    this->difference = difference;
}

void PathRelinking::differenceBetweenSolutions() {

    for (int i = 0; i < this->getProblem()->getCustomers(); ++i) {

        if (this->getInitialPositions().at(i) != this->getGuidePositions().at(i)) {
            this->getDifference().push_back(i);
        }

    }

}
