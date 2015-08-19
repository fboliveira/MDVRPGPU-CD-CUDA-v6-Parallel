/* 
 * File:   IndividualsGroup.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 23, 2014, 5:03 PM
 */

#include "IndividualsGroup.hpp"

/*
 * Constructors and Destructor
 */

IndividualsGroup::IndividualsGroup() {
    //this->setLocked(false);
    this->setLSProcessed(false);
}

IndividualsGroup::IndividualsGroup(MDVRPProblem* problem, AlgorithmConfig* config, int depot) :
problem(problem), config(config), depot(depot) {
    //this->setLocked(false);
    this->setLSProcessed(false);
}

IndividualsGroup::IndividualsGroup(const IndividualsGroup& orig) {

    this->setProblem(orig.getProblem());
    this->setConfig(orig.getConfig());
    this->setDepot(orig.getDepot());

    this->setIndividuals(orig.getIndividualsConst());

    this->setLocked(orig.isLocked());
    this->setLSProcessed(orig.isLSProcessed());

    this->setForceSequential(orig.isForceSequential());

}

/*
 * Getters and Setters
 */

int IndividualsGroup::getDepot() const {
    return this->depot;
}

void IndividualsGroup::setDepot(int depot) {
    this->depot = depot;
}

AlgorithmConfig* IndividualsGroup::getConfig() const {
    return this->config;
}

void IndividualsGroup::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* IndividualsGroup::getProblem() const {
    return this->problem;
}

void IndividualsGroup::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

vector<Individual>& IndividualsGroup::getIndividuals() {
    return this->individuals;
}

void IndividualsGroup::setIndividuals(vector<Individual> individuals) {
    this->individuals = individuals;
}

vector<Individual> IndividualsGroup::getIndividualsConst() const {
    return this->individuals;
}

bool IndividualsGroup::isLocked() const {
    return locked;
}

void IndividualsGroup::setLocked(bool locked) {

    //    if (this->isLocked() != locked) {
    //        if (locked)
    //            this->getConfig()->getMutexLocker().lock();
    //        else
    //            this->getConfig()->getMutexLocker().unlock();
    //    }

    this->locked = locked;
}

bool IndividualsGroup::isLSProcessed() const {
    return lsProcessed;
}

void IndividualsGroup::setLSProcessed(bool lsProcessed) {
    this->lsProcessed = lsProcessed;
}

bool IndividualsGroup::isForceSequential() const {
    return forceSequential;
}

void IndividualsGroup::setForceSequential(bool forceSequential) {
    this->forceSequential = forceSequential;
}

vector<CustomerPosition> IndividualsGroup::getCustomersPosition() {

    vector<CustomerPosition> position(this->getProblem()->getCustomers());

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [&position] (Individual & individual) {
        individual.setCustomersPosition(position);
    });

    return position;

}

vector<Rank>& IndividualsGroup::getRanks() {
    return ranks;
}

void IndividualsGroup::setRanks(vector<Rank> ranks) {
    this->ranks = ranks;
}

/*
 * Public Methods
 */

void IndividualsGroup::clear() {
    this->getIndividuals().clear();
}

void IndividualsGroup::add(Individual individual) {
    this->getIndividuals().push_back(individual);
}

size_t IndividualsGroup::size() {
    return this->getIndividuals().size();
}

size_t IndividualsGroup::getNumTotalCustomers() {

    size_t num = 0;

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [&num] (Individual & individual) {
        num += individual.getNumCustomers();
    });

    return num;

}

size_t IndividualsGroup::getNumTotalCustomersFromRoutes() {

    size_t num = 0;

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [&num] (Individual & individual) {
        num += individual.getNumCustomersRoute();
    });

    return num;

}

void IndividualsGroup::evaluate(bool removeConflicts, bool split) {

    if (removeConflicts)
        this->removeConflictedCustomersFromDepots();

    if (this->getNumTotalCustomersFromRoutes() != this->getProblem()->getCustomers())
        split = true;

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [split] (Individual & individual) {
        individual.evaluate(split);
    });

}

void IndividualsGroup::localSearch(bool fullImprovement) {

    int move, indU, indV, ru, rv;
    bool improved = false;

    if (fullImprovement)
        this->setForceSequential(true);

    try {

        do {

            improved = false;

            if (this->getProblem()->getMonitor().isTerminated())
                break;

            for (int indA = 0; indA < this->getIndividuals().size(); ++indA) {

                if (this->getProblem()->getMonitor().isTerminated())
                    break;

                if (!this->isForceSequential() && this->getConfig()->getLocalSearchType() == RANDOM)
                    indU = Random::randIntBetween(0, this->getIndividuals().size() - 1);
                else
                    indU = indA;

                for (int indB = 0; indB < this->getIndividuals().size(); ++indB) {

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;

                    if (!this->isForceSequential() && this->getConfig()->getLocalSearchType() == RANDOM)
                        indV = Random::randIntBetween(0, this->getIndividuals().size() - 1);
                    else
                        indV = indB;

                    //if (indU == indV)
                    //    continue;

                    for (int u = 0; u < this->getIndividuals().at(indU).getRoutes().size(); ++u) {

                        if (this->getProblem()->getMonitor().isTerminated())
                            break;

                        if (!this->isForceSequential() && this->getConfig()->getLocalSearchType() == RANDOM)
                            ru = Random::randIntBetween(0, this->getIndividuals().at(indU).getRoutes().size() - 1);
                        else
                            ru = u;

                        if (this->getIndividuals().at(indU).getRoutes().at(ru).getTour().size() == 0)
                            continue;

                        for (int v = 0; v < this->getIndividuals().at(indV).getRoutes().size(); ++v) {

                            if (this->getProblem()->getMonitor().isTerminated())
                                break;

                            if (!this->isForceSequential() && this->getConfig()->getLocalSearchType() == RANDOM)
                                rv = Random::randIntBetween(0, this->getIndividuals().at(indV).getRoutes().size() - 1);
                            else
                                rv = v;

                            if (this->getIndividuals().at(indV).getRoutes().at(rv).getTour().size() == 0)
                                continue;

                            vector<int> moves;
                            Random::randPermutation(this->getConfig()->getTotalMoves(), 1, moves);

                            for (int m = 1; m <= this->getConfig()->getTotalMoves(); ++m) {

                                if (this->getProblem()->getMonitor().isTerminated())
                                    break;

                                if (!this->isForceSequential() && this->getConfig()->getLocalSearchType() == RANDOM)
                                    move = moves.at(m - 1);
                                else
                                    move = m;

                                bool result;
                                bool equal = indU == indV && ru == rv;

                                do {
                                    result = LocalSearch::processMoveDepotRoute(this->getIndividuals().at(indU).getRoutes().at(ru),
                                            this->getIndividuals().at(indV).getRoutes().at(rv), move, equal);

                                    if (result)
                                        improved = true;

                                } while (result == true && !this->getProblem()->getMonitor().isTerminated());
                            }

                            if (this->getProblem()->getMonitor().isTerminated())
                                break;

                        }

                        this->getIndividuals().at(indV).routesToGenes();

                        if (this->getProblem()->getMonitor().isTerminated())
                            break;

                    }

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;

                }

                this->getIndividuals().at(indU).routesToGenes();

                if (this->getProblem()->getMonitor().isTerminated())
                    break;

            }

        } while (fullImprovement && improved && !this->getProblem()->getMonitor().isTerminated());

        this->setLSProcessed(true);
        this->setForceSequential(!improved);

    } catch (exception& e) {
        cout << e.what() << endl;
        this->printSolution();
        this->print();
    }

    //this->setLocked(false);

}

float IndividualsGroup::getTotalCost() {

    float cost = 0;

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [&cost] (Individual & individual) {
        cost += individual.getTotalCost();
    });

    cost += this->getIncompleteSolutionPenalty();
    return cost;

}

float IndividualsGroup::getIncompleteSolutionPenalty() {

    int nCustomers = this->getNumTotalCustomersFromRoutes();
    float penalty = 0.0;

    if (nCustomers != this->getProblem()->getCustomers())
        penalty = (abs(nCustomers - this->getProblem()->getCustomers()) * this->getConfig()->getIncompleteSolutionPenalty());

    return penalty;

}

bool IndividualsGroup::isChanged() {

    //    for(auto ite = this->getIndividuals().begin(); ite != this->getIndividuals().end(); ++ite) {
    //        
    //        if ((*ite).isChanged())
    //            return true;
    //        
    //    }

    return true;

}

bool IndividualsGroup::isPenalized() {

    bool penalized = false;

    for (auto ite = this->getIndividuals().begin(); ite != this->getIndividuals().end(); ++ite) {
        if ((*ite).isPenalized()) {
            penalized = true;
            break;
        }
    }

    if (this->getIncompleteSolutionPenalty() > 0.0)
        penalized = true;

    return penalized;

}

//void IndividualsGroup::merge(IndividualsGroup& source) {
//
//    //cout << "Size merge O = " << this->getIndividuals().size() << endl;
//    this->getIndividuals().insert(this->getIndividuals().begin(),
//            source.getIndividuals().begin(), source.getIndividuals().end());
//    //cout << "Size merge F = " << this->getIndividuals().size() << endl;
//
//}

void IndividualsGroup::rank(int source) {

    //    //cout << "Size sort 1 = " << this->getIndividuals().size() << endl;
    //    std::sort(this->getIndividuals().begin(), this->getIndividuals().end(), compareIndividuals);
    //    //this->printList();
    //    //cout << "Size sort 2 = " << this->getIndividuals().size() << endl;
    //
    //    for(int i = 0; i < this->getIndividuals().size(); ++i) {
    //        if (this->getIndividuals().at(i).getRoutes().empty())
    //            cout << "IndividualsGroup::sort() => empty: " << this->getIndividuals().at(i).getDepot() << "\t" << i 
    //                    << "\tCost: " << this->getIndividuals().at(i).getTotalCost() <<  endl;
    //    }

    this->getRanks().clear();
    for (int i = 0; i < this->getIndividuals().size(); ++i) {
        Rank rnk = Rank(source, i, this->getIndividuals().at(i).getTotalCost());
        this->getRanks().push_back(rnk);
    }

    //std::sort(this->getRanks().begin(), this->getRanks().end(), Rank::compare);

}

void IndividualsGroup::shrink(IndividualsGroup& source) {

    //    this->getIndividuals().erase( this->getIndividuals().begin() + this->getConfig()->getNumSubIndDepots(), this->getIndividuals().end());
    //    //cout << "Size shrink = " << this->getIndividuals().size() << endl;
    //    
    //    for(int i = 0; i < this->getIndividuals().size(); ++i) {
    //        this->getIndividuals().at(i).setId(i);
    //        if (this->getIndividuals().at(i).getRoutes().empty())
    //            cout << "IndividualsGroup::shrink() => empty: " << this->getDepot() << "\t" << i << endl;
    //    }

    //printf("IndividualsGroup::shrink() => %d\n", this->getDepot());

    vector<float> cost;

    this->rank(0);
    source.rank(1);

    this->getRanks().insert(this->getRanks().end(), source.getRanks().begin(), source.getRanks().end());
    std::sort(this->getRanks().begin(), this->getRanks().end(), Rank::compare);

    //for (int i = 0; i < this->getIndividuals().size(); ++i) {

    int i = 0, w = 0;
    while (w < this->getIndividuals().size()) {

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        auto ite = std::find(cost.begin(), cost.end(),
                Util::scaledFloat(this->getRanks().at(i).getCost()));

        if (ite == cost.end()) {

            if (this->getConfig()->getProcessType() == Enum_Process_Type::MULTI_THREAD)
                this->getProblem()->getMonitor().getLock(this->getDepot(), w)->wait(false);

            int id = this->getRanks().at(i).getId();
            this->getIndividuals().at(w).setId(w); // New id

            if (this->getRanks().at(i).getSource() == 0) {
                this->getIndividuals().at(w).setGene(this->getIndividuals().at(id).getGene());
                this->getIndividuals().at(w).setRoutes(this->getIndividuals().at(id).getRoutesConst());
            } else {
                this->getIndividuals().at(w).setGene(source.getIndividuals().at(id).getGene());
                this->getIndividuals().at(w).setRoutes(source.getIndividuals().at(id).getRoutesConst());
            }

            cost.push_back(Util::scaledFloat(this->getRanks().at(i).getCost()));
            w++;

        }

        i++;
                
    }


}

void IndividualsGroup::print() {

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [] (Individual & individual) {
        individual.print();
    });

}

void IndividualsGroup::print(bool gene) {

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [] (Individual & individual) {
        individual.print(true);
    });

}

void IndividualsGroup::printSolution() {

    printf("\n%.2f", this->getTotalCost());

    if (this->isPenalized())
        printf("\tInfeasible");

    printf("\n");

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [] (Individual & individual) {
        individual.printSolution();
    });

    printf("\n\n");
    
    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [] (Individual & individual) {
        individual.printVehicles();
    });
    
}

void IndividualsGroup::printList() {

    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [] (Individual & individual) {
        printf("Id = %d\t%.2f\n", individual.getId(), individual.getTotalCost());
    });

}

/*
 * Private Methods
 */

void IndividualsGroup::removeConflictedCustomersFromDepots() {

    int customer, d, n, bestDep, pos, i, j;
    float cost, saveCost, costIJ;

    // Get all attendances for each customers
    for (customer = 0; customer < this->getProblem()->getCustomers(); ++customer) {

        n = 0;
        vector<int> attendances(this->getProblem()->getDepots(), 0);

        // Calculate the cost of customer in each depot
        for (d = 0; d < this->getProblem()->getDepots(); ++d) {

            if (this->getIndividuals().at(d).getNumCustomers() > 0) {

                this->getIndividuals().at(d).removeRepeteadCustomer(customer + 1);

                pos = this->getIndividuals().at(d).find(customer + 1);

                if (pos >= 0) {
                    attendances.at(d) = 1;
                    n++;
                } else {
                    attendances.at(d) = 0;
                }
            }
        }

        // If there is no attendance
        if (n == 0) {

            // Repair solution
            // Check the best depot -- according allocation - global memory
            /*d = -1;
            while (d < 0) {
                d = Random::randIntBetween(0, this->getProblem()->getDepots() - 1);
                if (this->getProblem()->getAllocation().at(customer).at(d) == 0)
                    d = -1;
            }*/
            
            typedef_location bestLocation;
            bestLocation.cost = FLT_MAX;
            
            for(d = 0; d< this->getProblem()->getDepots();++d) {
                if (this->getProblem()->getAllocation().at(customer).at(d) > 0) {
                    typedef_location location = this->getIndividuals().at(d).getMinimalPositionToInsert(customer + 1);
                    
                    if (location.cost < bestLocation.cost) {
                        bestLocation.depot = d;
                        bestLocation.cost = location.cost;
                        bestLocation.position = location.position;
                    }
                }
            }
            
            this->getIndividuals().at( bestLocation.depot ).add(bestLocation.position, customer + 1);

        }

        if (n > 1) {

            saveCost = FLT_MAX;
            bestDep = -1;

            for (d = 0; d < this->getProblem()->getDepots(); ++d) {

                cost = 0;
                costIJ = 0;

                if (attendances.at(d) == 0)
                    continue;

                // Get the position in depot
                pos = this->getIndividuals().at(d).find(customer + 1);

                if (pos < 0)
                    continue;

                if (this->getIndividuals().at(d).getNumCustomers() == 1)
                    cost = FLT_MAX;
                else {

                    //i - before
                    //k - current customer
                    //j - after

                    // Best Saving = Min[(Cik + Ckj) - Cij]
                    // I - K - J or I - J ?
                    // Se o valor Cij for alto, indica que eh melhor manter Cjk+Ckj

                    // Cik
                    if (pos == 0) {
                        // D - C
                        cost = this->getProblem()->getDepotDistances().at(d).at(customer);
                        // I - D(j)
                        i = this->getIndividuals().at(d).getCustomer(pos + 1) - 1;

                        if (i > this->getProblem()->getCustomers()) {
                            cout << "removeConflictedCustomersFromDepots: Erro!\n";
                            this->getIndividuals().at(d).print();
                        }

                        costIJ = this->getProblem()->getDepotDistances().at(d).at(i);
                    } else {
                        // C-1 - C
                        i = this->getIndividuals().at(d).getCustomer(pos - 1) - 1;

                        if (i > this->getProblem()->getCustomers()) {
                            cout << "removeConflictedCustomersFromDepots: Erro!\n";
                            this->getIndividuals().at(d).print();
                        }

                        cost = this->getProblem()->getCustomerDistances().at(i).at(customer);
                    }

                    // Ckj
                    if (pos == this->getIndividuals().at(d).getNumCustomers() - 1)
                        // C - D
                        cost += this->getProblem()->getDepotDistances().at(d).at(customer);
                    else
                        // C - C+1
                        cost += this->getProblem()->getCustomerDistances().at(customer).at(this->getIndividuals().at(d).getCustomer(pos + 1) - 1);

                    // Cij
                    if (pos > 0) {
                        // I -- before k
                        i = this->getIndividuals().at(d).getCustomer(pos - 1) - 1;

                        if (pos == this->getIndividuals().at(d).getNumCustomers() - 1)
                            // J = D
                            costIJ = this->getProblem()->getDepotDistances().at(d).at(i);
                        else {
                            // J = k + 1
                            j = this->getIndividuals().at(d).getCustomer(pos + 1) - 1;
                            costIJ = this->getProblem()->getCustomerDistances().at(i).at(j);
                        }
                    }
                }

                cost -= costIJ;

                // Best saving
                if (cost < saveCost) {
                    saveCost = cost;
                    bestDep = d;
                }
            }

            // Remove customer from other depots
            for (d = 0; d < this->getProblem()->getDepots(); ++d) {

                if (attendances.at(d) == 0)
                    continue;

                if (d != bestDep)
                    //removeCustomerFromIndividual(&individuals->depot.at(d), customer + 1);
                    this->getIndividuals().at(d).remove(customer + 1);

            }

        }
    }

    if (this->getNumTotalCustomers() != this->getProblem()->getCustomers()) {

        vector<int> rep(this->getProblem()->getCustomers(), 0);

        for (d = 0; d < this->getProblem()->getDepots(); ++d)
            for (int c = 0; c < this->getIndividuals().at(d).getNumCustomers(); ++c) {
                rep.at(this->getIndividuals().at(d).getCustomer(c) - 1)++;

                if (rep.at(this->getIndividuals().at(d).getCustomer(c) - 1) > 1)
                    cout << "Customer: " << this->getIndividuals().at(d).getCustomer(c) << "\n";
            }

        this->print();

        //Util::print(rep.begin(), rep.end());

    }

}

bool IndividualsGroup::compareIndividuals(Individual i, Individual j) {
    return Util::isBetterSolution(i.getTotalCost(), j.getTotalCost());
}
