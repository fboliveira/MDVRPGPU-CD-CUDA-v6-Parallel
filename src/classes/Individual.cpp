/*
 * File:   Individual.cpp
 * Author: fernando
 *
 * Created on July 21, 2014, 2:05 PM
 */

#include "Individual.hpp"

/*
 * Constructors
 */

Individual::Individual() {
    this->setDepot(-1);
    this->setId(-1);
}

Individual::Individual(MDVRPProblem* problem, AlgorithmConfig* config, int depot, int id) :
problem(problem), config(config), depot(depot), id(id) { //, mutexLocker(mutexLocker), conditionVariable(conditionVariable) {

    this->setChanged(true);
    this->setLocked(false);

    this->setMutationRatePM(this->getConfig()->getMutationRatePM());
    this->setMutationRatePLS(this->getConfig()->getMutationRatePLS());

    this->setRelaxSplit(false);
    this->setRestartMoves(false);

}

// Copy constructor
Individual::Individual(const Individual& other) :
problem(other.problem), config(other.config), depot(other.depot),
id(other.id), changed(other.changed), gene(other.gene),
routes(other.routes),
locked(other.locked), numOperations(other.numOperations),
mutationRatePM(other.mutationRatePM), mutationRatePLS(other.mutationRatePLS),
relaxSplit(other.relaxSplit), restartMoves(other.restartMoves) {
}

/*
 * Getters and Setters
 */

vector<int>& Individual::getGene() {
    return gene;
}

vector<int> Individual::getGeneConst() const {
    return this->gene;
}

void Individual::setGene(vector<int> gene) {
    this->gene = gene;
}

vector<Route>& Individual::getRoutes() {
    return routes;
}

vector<Route> Individual::getRoutesConst() const {
    return this->routes;
}

void Individual::setRoutes(vector<Route> routes) {
    this->routes = routes;
}

int Individual::getId() const {
    return this->id;
}

void Individual::setId(int id) {
    this->id = id;
}

int Individual::getDepot() const {
    return this->depot;
}

void Individual::setDepot(int depot) {
    this->depot = depot;
}

AlgorithmConfig* Individual::getConfig() const {
    return this->config;
}

void Individual::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* Individual::getProblem() const {
    return this->problem;
}

void Individual::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

bool Individual::isChanged() const {
    return this->changed;
}

void Individual::setChanged(bool changed) {
    this->changed = changed;
}

size_t Individual::getNumCustomers() {
    return this->getGene().size();
}

size_t Individual::getNumCustomersRoute() {

    size_t customers = 0;

    for (auto ite = this->getRoutes().begin(); ite != this->getRoutes().end(); ++ite) {
        customers += (*ite).getTour().size();
    }

    return customers;
}

int Individual::getCustomer(int position) {
    return this->getGene().at(position);
}

bool Individual::isLocked() const {
    return locked;
}

void Individual::setLocked(bool locked) {

    //    if (locked)
    //        this->getConfig()->getMutexLocker().lock();
    //    else
    //        this->getConfig()->getMutexLocker().unlock();

    this->locked = locked;
}

int Individual::getNumOperations() const {
    return numOperations;
}

void Individual::setNumOperations(int numOperations) {
    this->numOperations = numOperations;
}

float Individual::getMutationRatePM() const {
    return mutationRatePM;
}

void Individual::setMutationRatePM(float mutationRatePM) {
    this->mutationRatePM = mutationRatePM;
}

float Individual::getMutationRatePLS() const {
    return mutationRatePLS;
}

void Individual::setMutationRatePLS(float mutationRatePLS) {
    this->mutationRatePLS = mutationRatePLS;
}

bool Individual::isRelaxSplit() const {
    return relaxSplit;
}

void Individual::setRelaxSplit(bool relaxSplit) {
    this->relaxSplit = relaxSplit;
}

bool Individual::isRestartMoves() const {
    return restartMoves;
}

void Individual::setRestartMoves(bool restartMoves) {
    this->restartMoves = restartMoves;
}

//std::mutex Individual::getMutexLocker() const {
//    return mutexLocker;
//}

//void Individual::setMutexLocker(std::mutex mutexLocker) {
//    this->mutexLocker = mutexLocker;
//}

//std::condition_variable Individual::getConditionVariable() const {
//    return conditionVariable;
//}

//void Individual::setConditionVariable(std::condition_variable conditionVariable) {
//    this->conditionVariable = conditionVariable;
//}

void Individual::setCustomersPosition(vector<CustomerPosition>& position) {

    this->updateRoutesID();

    for_each(this->getRoutes().begin(), this->getRoutes().end(), [&position](Route & route) {
        route.setCustomersPosition(position);
    });

}

/*
 * Public Methods
 */

void Individual::add(int customer) {

    this->getGene().push_back(customer);
    this->setChanged(true);

}

//template<typename Iter>

void Individual::add(typedef_vectorIntIterator position, int customer) {

    this->getGene().insert(position, customer);
    this->setChanged(true);

}

void Individual::remove(int customer) {

    this->getGene().erase(std::remove(this->getGene().begin(), this->getGene().end(), customer), this->getGene().end());
    this->setChanged(true);

}

template<typename Iter>
void Individual::remove(Iter position) {

    this->getGene().erase(position);
    this->setChanged(true);

}

int Individual::find(int customer) {

    auto it = std::find(this->getGene().begin(), this->getGene().end(), customer);

    if (it == this->getGene().end())
        return -1;
    else
        return it - this->getGene().begin();

    //return Util::findValueInVector(this->getGene().begin(), this->getGene().end(), customer);
}

void Individual::create() {

    this->generateInitialSolutionRandomNearestInsertion();
    this->setNumOperations(Random::randIntBetween(1, this->getGene().size()));
    //this->evaluate(true);

}

void Individual::evaluate(bool split) {

    /*Self-adaptive*/
    if (this->isPenalized()) {
        this->setRelaxSplit(true);
        split = true;
    }
    else
        this->setRelaxSplit(false);

    if (split || this->isChanged()) {
        this->split();
    }
    else {

        //        for_each(this->getRoutes().begin(), this->getRoutes().end(), [] (Route & route) {
        //            
        //            if (route.getTour().size() == 0)
        //                cout << "Erro!\n";
        //            
        //            route.calculateCost();
        //        });

        auto ite = this->getRoutes().begin();

        while (ite != this->getRoutes().end()) {
            if ((*ite).getTour().empty())
                ite = this->getRoutes().erase(ite);
            else {
                (*ite).calculateCost();
                ite++;
            }
        }

    }

    this->setChanged(false);

}

Individual Individual::evolve() {

    //    //If it is locked
    //    if (this->isLocked())
    //        return Individual();

    this->setLocked(true);

    Individual offspring = this->copy(true);

    if (this->getGene().size() == 0) {
        printf("Subpop = %d => Ind vazio!\n", this->getDepot());
        Individual offspring = Individual(this->getProblem(), this->getConfig(), this->getDepot(), this->getId());
        offspring.create();
    }

    //cout << "Offspring = " << offspring.getTotalCost() << endl;

    bool mutate = false;

    //printf("Offspring = D: %d - Id: %d => PM: %.2f / PLS: %.2f / RS: %d\n", this->getDepot(), 
    //        this->getId(), this->getMutationRatePM(), this->getMutationRatePLS(), this->isRestartMoves());

    if (Random::randFloat() <= offspring.getMutationRatePM()) {
        offspring.mutate();
        offspring.evaluate(true);
        mutate = true;
    }
    else {

        if (offspring.getRoutes().size() == 0)
            offspring.evaluate(true);

    }

    //offspring.printSolution();

    if (Random::randFloat() <= offspring.getMutationRatePLS()) {

        //Individual orig = offspring;
        //offspring.printSolution(true);

        offspring.localSearch();
        offspring.routesToGenes();

        //offspring.printSolution(true);

        if (offspring.getGene().size() == 0) {

            cout << "Mutate = " << mutate << endl;

            this->print();
            this->print(true);
            this->printSolution();

            offspring.print();
            offspring.print(true);
            offspring.printSolution();

            //            orig.print();
            //            orig.print(true);
            //            orig.printSolution();

        }

    }

    if (offspring.getRoutes().size() == 0) {
        //cout << "Wait\n\n";
        //cout << "Old: " << this->getTotalCost() << endl;
        offspring.evaluate(true);
        //cout << "New: " << this->getTotalCost() << endl;
    }

    // Update parent parameter, if there is no improvement.
    if (Util::isBetterSolution(offspring.getTotalCost(), this->getTotalCost())) {
        this->updateParameters(true);
    }
    else {
        this->updateParameters(false);
    }

    //this->print(true);    
    //this->setLocked(false);

    return offspring;

}

//void Individual::mutate() {
//
//    // Mutation
//    int first, last;
//
//    first = Random::randIntBetween(0, this->getGene().size() - 1);
//    last = first;
//
//    while (last == first) {
//        last = Random::randIntBetween(0, this->getGene().size() - 1);
//    }
//
//    if (first > last)
//        swap(first, last);
//
//    rotate(this->getGene().begin() + first, this->getGene().begin() + last, this->getGene().end());
//    this->setChanged(true);
//
//}

void Individual::mutate() {

    int x, y;

    this->autoUpdate(true);

    try {
        for (int i = 0; i < this->getNumOperations(); ++i) {
            Random::randTwoNumbers(0, this->getGene().size() - 1, x, y);
            swap(this->getGene().at(x), this->getGene().at(y));
        }
    }
    catch (exception &e) {
        cout << "Individual::mutate() => " << e.what() << '\n';
        cout << "X = " << x << "\tY = " << y << "\tSize = " << this->getGene().size() << endl;
    }

    this->setChanged(true);
}

void Individual::localSearch() {

    int move, ru, rv;

    //try {

        for (int u = 0; u < this->getRoutes().size(); ++u) {

            if (this->getProblem()->getMonitor().isTerminated())
                break;

            if (this->getConfig()->getLocalSearchType() == RANDOM)
                ru = Random::randIntBetween(0, this->getRoutes().size() - 1);
            else
                ru = u;

            if (this->getRoutes().at(ru).getTour().size() == 0) {
                continue;
            }

            for (int v = 0; v < this->getRoutes().size(); ++v) {

                if (this->getProblem()->getMonitor().isTerminated())
                    break;

                if (this->getConfig()->getLocalSearchType() == RANDOM)
                    rv = Random::randIntBetween(0, this->getRoutes().size() - 1);
                else
                    rv = v;

                if (this->getRoutes().at(rv).getTour().size() == 0)
                    continue;

                vector<int> moves;
                Random::randPermutation(this->getConfig()->getTotalMoves(), 1, moves);

                for (int m = 1; m <= this->getConfig()->getTotalMoves(); ++m) {

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;

                    if (this->getConfig()->getLocalSearchType() == RANDOM)
                        move = moves.at(m - 1);
                    else
                        move = m;

                    bool result = false;
                    do {
                        result = LocalSearch::processMoveDepotRoute(this->getRoutes().at(ru), this->getRoutes().at(rv),
                            move, ru == rv, false, -1);

                    } while (result && this->isRestartMoves() && !this->getProblem()->getMonitor().isTerminated());

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;
                }

                if (this->getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (this->getProblem()->getMonitor().isTerminated())
                break;

        }
    //}
    //catch (exception &e) {
    //    cout << "Individual::localSearch(): " << e.what() << '\n';
    //}

    this->setChanged(false);

}

float Individual::getTotalCost() {

    float cost = 0.0;

    for_each(this->getRoutes().begin(), this->getRoutes().end(), [&cost](Route & route) {
        cost += route.getTotalCost();
    });

    cost += this->getPenaltyVehicles();

    if (cost <= 0)
        cost = FLT_MAX;

    return cost;
}

float Individual::getPenaltyVehicles() {

    float penalty = 0.0;

    // Relax penalty when split is relaxed
    if (!this->isRelaxSplit()) {

        int validRoutes = getNumVehicles();

        if (validRoutes > this->getProblem()->getVehicles())
            penalty = this->getConfig()->getExtraVehiclesPenalty() * (validRoutes - this->getProblem()->getVehicles());
    }

    return penalty;

}

int Individual::getNumVehicles() {

    int validRoutes = 0;

    for (auto ite = this->getRoutes().begin(); ite != this->getRoutes().end(); ++ite)
        if (!(*ite).getTour().empty())
            validRoutes++;

    return validRoutes;

}

bool Individual::isPenalized() {

    bool penalized = false;

    for (auto ite = this->getRoutes().begin(); ite != this->getRoutes().end(); ++ite)
        if ((*ite).isPenalized()) {
            penalized = true;
            break;
        }

    if (this->getPenaltyVehicles() > 0.0)
        penalized = true;

    return penalized;

}

/*
 *  Description: Split an entire route using optimal splitting procedure - Prins(2004)
 *   - Feasible solutions are created in terms of capacity (W) and the
 *   number of vehicles (m)
 */

void Individual::split() {

    this->getRoutes().clear();
    this->setChanged(true);

    int i, j, c, load, nv;
    float cost, routeDuration;
    int capacity;

    //this->print();

    // Infinite
    routeDuration = FLT_MAX;

    if (this->getProblem()->getDuration() > 0) {
        if (this->isRelaxSplit())
            routeDuration = 2 * this->getProblem()->getDuration();
        else
            routeDuration = this->getProblem()->getDuration();
    }

    capacity = this->getProblem()->getCapacity();

    if (this->getGene().size() == 0)
        return;

    int length = this->getGene().size();

    vector<int> P(length + 1, 0);
    vector<float> V(length + 1, FLT_MAX);

    V.at(0) = 0;

    //imprimirVector(route);
    for (i = 0; i < length; ++i) {

        load = 0;
        cost = 0;
        nv = 0;
        j = i;

        do {

            c = this->getGene().at(j) - 1;

            if (c < 0 || c > this->getProblem()->getCustomers()) {
                Util::error("splitRoute:C: Valor invalido", j);
                //Util::print(this->getGene().begin(), this->getGene().end());
                return;
                //return trip;
            }

            load += this->getProblem()->getDemand().at(c);

            if (i == j) {
                cost = this->getProblem()->getDepotDistances().at(this->getDepot()).at(c) * 2;
                // Compute service time
                cost += this->getProblem()->getServiceTime().at(c);
            }
            else {

                if (this->getGene().at(j - 1) <= 0 || this->getGene().at(j - 1) > this->getProblem()->getCustomers()) {
                    Util::error("splitRoute:J-1: Valor invalido", j);
                    //Util::print(this->getGene().begin(), this->getGene().end());
                    return;
                    //return trip;
                }

                // cost = cost - c(j, 1) + c(j, j+1) + c(j+1,1);
                // - custo do anterior ate o deposito
                // + custo do anterior ate o corrente
                // + custo do corrente ate o deposito                
                cost = cost - this->getProblem()->getDepotDistances().at(this->getDepot()).at(this->getGene().at(j - 1) - 1)
                    + this->getProblem()->getCustomerDistances().at(this->getGene().at(j - 1) - 1).at(c)
                    + this->getProblem()->getDepotDistances().at(this->getDepot()).at(c);

                // Compute service time
                cost += this->getProblem()->getServiceTime().at(c);
                //cost -= this->getProblem()->getServiceTime().at(this->getGene().at(j - 1) - 1);

            }

            if (load <= capacity && cost <= routeDuration) {
                if (V.at(i) + cost < V.at(j + 1)) {
                    V.at(j + 1) = V.at(i) + cost;
                    P.at(j + 1) = i;
                }
                j++;
            }

            // Until            
        } while ((j < length) && (load <= capacity) && (cost <= routeDuration));
    }

    this->splitExtract(P);

}

void Individual::routesToGenes() {

    //this->getGene().clear();

    vector<int> genes;

    //    for (auto position = this->getRoutes().begin(); position != this->getRoutes().end(); ++position) {
    //
    //        if ((*position).getTour().size() == 0)
    //            this->getRoutes().erase(position);
    //        else
    //            for_each((*position).getTour().begin(), (*position).getTour().end(), [&genes] (int customer) {
    //                genes.push_back(customer);
    //            });
    //    }

    for_each(this->getRoutes().begin(), this->getRoutes().end(), [&genes](Route & route) {

        //std::copy(route.getTour().begin(), route.getTour().end(), genes.end());

        for_each(route.getTour().begin(), route.getTour().end(), [&genes](int customer) {
            genes.push_back(customer);
        });

    });

    //cout << endl;

    //    if (genes.size() == 0) {
    //        cout << "Nenhum gene!\n";
    //        this->printSolution();
    //        this->print();
    //        this->print(true);
    //    } else {
    //        this->setGene(genes);
    //    }

    if (genes.size() != 0)
        this->setGene(genes);
    //else
    //    this->create();

    //this->print(true);
    //this->printSolution();

}



int Individual::autoUpdate(bool update) {

    int max = (int) this->getGene().size();
    int op = Random::discreteDistribution(1, max);

    if (update)
        this->setNumOperations(op);

    return op;

}

void Individual::updateParameters(bool improved) {

    if (!improved) {
        //this->setMutationRatePLS(this->getMutationRatePLS() + 0.1);
        //this->setMutationRatePM(this->getMutationRatePM() + 0.05);
        this->setRestartMoves(!this->isRestartMoves());
    }
    //
    //    if (this->getMutationRatePLS() > 0.9)
    //        this->setMutationRatePLS(0.1);
    //
    //    //if (this->getMutationRatePM() > 0.9)
    //    //    this->setMutationRatePM(0.1);

}

void Individual::removeRepeteadCustomer(int customer) {

    // Count the number of customer on the route
    int count = std::count(this->getGene().begin(), this->getGene().end(), customer);

    if (count > 1) {

        this->remove(customer);
        // Call recursively
        this->removeRepeteadCustomer(customer);

    }

}

// Description: Get minimal point to insert a customer on a route
// Best saving

typedef_location Individual::getMinimalPositionToInsert(int customer) {

    typedef_location bestLocation;

    //template<typename Iter>
    //Iter Individual::getMinimalPositionToInsert(Iter position, int customer) {

    // If there is no customer on the route
    if (this->getNumCustomers() == 0) {
        // First position
        bestLocation.cost = 0;
        bestLocation.position = this->getGene().end();
        return bestLocation;
    }

    bestLocation.position = this->getGene().begin();
    customer--;

    //Iter position = this->getGene().begin(); //, p, i;
    float costCij, costCikj, localSaving;

    // (Cik + Ckj) - Cij --> Saving
    // Best min(Saving)

    // From Depot To Customer 0
    // Cij = Dep -> C0
    try {
        costCij = this->getProblem()->getDepotDistances().at(this->getDepot()).at(this->getGene().at(0) - 1);
    }
    catch (exception &e) {
        cout << "Individual::getMinimalPositionToInsert => " << e.what() << '\n';
        cout << this->getDepot() << endl;
        this->printSolution(true);
    }
    // Cik + Ckj = Dep -> Ck -> C0
    costCikj = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer) +
        this->getProblem()->getCustomerDistances().at(customer).at(this->getGene().at(0) - 1);

    // Best saving until now
    localSaving = costCikj - costCij;
    bestLocation.cost = localSaving;

    for (auto i = this->getGene().begin(); i != this->getGene().end(); ++i) {

        if (next(i) == this->getGene().end())
            break;

        // Cij = Ci -> Ci+1
        costCij = this->getProblem()->getCustomerDistances().at((*i) - 1).at((*next(i)) - 1);
        // Cik + Ckj = Ci -> Ck -> Ci + 1;
        costCikj = this->getProblem()->getCustomerDistances().at((*i) - 1).at(customer) +
            this->getProblem()->getCustomerDistances().at(customer).at((*next(i)) - 1);

        localSaving = costCikj - costCij;

        if (localSaving < bestLocation.cost) {
            bestLocation.position = next(i);
            bestLocation.cost = localSaving;
        }

    }

    // From the last Customer to Depot
    size_t last = this->getGene().size() - 1;
    // Cij = Cn -> Dep
    costCij = this->getProblem()->getDepotDistances().at(this->getDepot()).at(this->getGene().at(last) - 1);
    // Cik + Ckj = Cn -> Ck -> Dep
    costCikj = this->getProblem()->getCustomerDistances().at(this->getGene().at(last) - 1).at(customer) +
        this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer);

    localSaving = costCikj - costCij;

    if (localSaving < bestLocation.cost) {

        bestLocation.position = this->getGene().end();
        bestLocation.cost = localSaving;
    }

    return bestLocation;


}

//typedef_location Individual::getMinimalPositionToInsert(int customer) {
//
//    typedef_location bestLocation;
//    
//    //template<typename Iter>
//    //Iter Individual::getMinimalPositionToInsert(Iter position, int customer) {
//
//    // If there is no customer on the route
//    if (this->getNumCustomers() == 0)
//        // First position
//        return this->getGene().end();
//
//    typedef_vectorIntIterator position = this->getGene().begin();
//    customer--;
//
//    //Iter position = this->getGene().begin(); //, p, i;
//    float costCij, costCikj, bestSaving, localSaving;
//
//    // (Cik + Ckj) - Cij --> Saving
//    // Best min(Saving)
//
//    // From Depot To Customer 0
//    // Cij = Dep -> C0
//    try {
//        costCij = this->getProblem()->getDepotDistances().at(this->getDepot()).at(this->getGene().at(0) - 1);
//    } catch (exception &e) {
//        cout << "Individual::getMinimalPositionToInsert => " << e.what() << '\n';
//        cout << this->getDepot() << endl;
//        this->printSolution(true);
//    }
//    // Cik + Ckj = Dep -> Ck -> C0
//    costCikj = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer) +
//            this->getProblem()->getCustomerDistances().at(customer).at(this->getGene().at(0) - 1);
//
//    // Best saving until now
//    localSaving = costCikj - costCij;
//    bestSaving = localSaving;
//
//    for (auto i = this->getGene().begin(); i != this->getGene().end(); ++i) {
//
//        if (next(i) == this->getGene().end())
//            break;
//
//        // Cij = Ci -> Ci+1
//        costCij = this->getProblem()->getCustomerDistances().at((*i) - 1).at((*next(i)) - 1);
//        // Cik + Ckj = Ci -> Ck -> Ci + 1;
//        costCikj = this->getProblem()->getCustomerDistances().at((*i) - 1).at(customer) +
//                this->getProblem()->getCustomerDistances().at(customer).at((*next(i)) - 1);
//
//        localSaving = costCikj - costCij;
//
//        if (localSaving < bestSaving) {
//            position = next(i);
//            bestSaving = localSaving;
//        }
//
//    }
//
//    // From the last Customer to Depot
//    size_t last = this->getGene().size() - 1;
//    // Cij = Cn -> Dep
//    costCij = this->getProblem()->getDepotDistances().at(this->getDepot()).at(this->getGene().at(last) - 1);
//    // Cik + Ckj = Cn -> Ck -> Dep
//    costCikj = this->getProblem()->getCustomerDistances().at(this->getGene().at(last) - 1).at(customer) +
//            this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer);
//
//    localSaving = costCikj - costCij;
//
//    if (localSaving < bestSaving) {
//
//        position = this->getGene().end();
//        bestSaving = localSaving;
//    }
//
//    return position;
//
//
//}

Individual Individual::copy(bool data) {

    Individual individual = Individual(this->getProblem(), this->getConfig(), this->getDepot(), this->getId());

    individual.setProblem(this->getProblem());
    individual.setConfig(this->getConfig());

    individual.setId(this->getId());
    individual.setDepot(this->getDepot());
    individual.setChanged(this->isChanged());

    if (data) {
        individual.setGene(this->getGene());
        individual.setRoutes(this->getRoutesConst());
    }

    individual.setLocked(this->isLocked());

    individual.setNumOperations(this->getNumOperations());

    individual.setMutationRatePM(this->getMutationRatePM());
    individual.setMutationRatePLS(this->getMutationRatePLS());
    individual.setRelaxSplit(this->isRelaxSplit());
    individual.setRestartMoves(this->isRestartMoves());

    return individual;

}

void Individual::update(Individual& source) {

    if (!this->isLocked())
        return;

    if (Util::isBetterSolution(source.getTotalCost(), this->getTotalCost())) {
        this->setGene(source.getGene());
        this->setRoutes(source.getRoutesConst());
        this->updateParameters(true);
    }

}

void Individual::print() {

    cout << "Dep = " << this->getDepot() << " - ID = " << this->getId() << " => Cost: " << this->getTotalCost() << endl;

    for_each(this->getRoutes().begin(), this->getRoutes().end(), [](Route & route) {

        route.print();
    });

}

void Individual::print(bool gene) {

    this->getProblem()->getMonitor().getMutexLocker().lock();
    cout << "Dep: " << this->getDepot() << "  Id: " << this->getId() << " -> Cost: " << this->getTotalCost() << " => ";
    Util::print(this->getGene());
    this->getProblem()->getMonitor().getMutexLocker().unlock();

}

void Individual::printSolution(bool insertTotalCost) {

    //}

    //void Individual::printSolution() {

    this->updateRoutesID();

    if (insertTotalCost)
        cout << "Dep = " << this->getDepot() << " - ID = " << this->getId() << " => Cost: " << this->getTotalCost() << endl;

    int i = 0;

    for_each(this->getRoutes().begin(), this->getRoutes().end(), [&i](Route & route) {
        route.setId(i);
        route.printSolution();
        i++;
    });

}

void Individual::printVehicles() {

    cout << "Dep = " << this->getDepot() << " - ID = " << this->getId() << " => Cost: " << this->getTotalCost() << "\t";
    cout << "Veh: " << this->getNumVehicles() << " / " << this->getProblem()->getVehicles() << endl;

}


/*
 * Private Methods
 */

/*
 *  Description: Generate initial solutions
 *
 *  This process is basead on Nearest Insertion, except that node i is
 *  selected randomly and it is allocate on current depot.
 *
 *  AVALIATION (COST) => DISTANCE
 *
 */
void Individual::generateInitialSolutionRandomNearestInsertion() {

    // Get allocated customers on depot from global memory
    int length, nClosest, i, nRCL, k, p, count = 0;

    bool greedy = this->getId() == 0;

    vector<int> closest;
    vector<int> allocated;

    this->getProblem()->getDepotGroup(this->getDepot(), allocated);
    length = allocated.size();

    // Anyone customer was allocated
    if (length == 0)
        return;

    // 1 - Select a node i to start
    i = 0;

    // If a greedy process
    if (greedy)
        // Select the nearest customer from depot
        i = allocated.at(0);
    else {
        // Select a random customer to start
        i = Random::randIntBetween(0, length - 1);
        // Get correct index
        i = allocated.at(i);
    }

    // Insert customer i
    this->add(i + 1);
    count++;

    // While there are customers
    while (count < length) {

        // List of the closest ones (from allocated)
        closest.clear();
        this->getProblem()->getNearestCustomerFromCustomerOnDepot(i, this->getDepot(), closest);
        nClosest = (int)closest.size();

        //imprimirVetor(closest, nClosest, 0);

        // restricted candidate list (RCL) = % num depots
        nRCL = nClosest / this->getProblem()->getDepots();

        k = -1;
        p = 0;

        while (k == -1) {

            // If a greedy process
            if (greedy)
                // Get the closest that is not served
                k = closest.at(p);
            else {
                // If the number of tries is bigger than the size of RCL,
                // the closest is increased to the limit of customers.
                if (p > nRCL)
                    nRCL = length;

                // Get a random customer with nRCL - semi-greedy
                k = -1;
                while (k == -1) {
                    k = Random::randIntBetween(0, nRCL - 1);
                    k = closest.at(k);

                    // If customer is not allocated in the depot
                    if (this->getProblem()->getAllocation().at(k).at(this->getDepot()) == 0)
                        k = -1;
                }

            }

            // If customer is served
            //if (existeValorVetor(depot->gene, k + 1) >= 0) {
            if (this->find(k + 1) >= 0) {

                k = -1;
                p = p + 1;
            }
        }

        // Insert customer k
        this->add(k + 1);
        count++;
        // The current (last) customer is k now
        i = k;

    }
}

void Individual::splitExtract(vector<int>& P) {
    //this->splitExtractRouteLimited(P);
    this->splitExtractWithoutVehicleLimit(P);
}

void Individual::splitExtractWithoutVehicleLimit(vector<int>& P) {

    int n = this->getGene().size();

    int p = 0, t = 0, j = n, i, k, id = 0;

    // Depot
    //--trip.push_back(ROUTE_DELIM);
    p++;

    // Repeat
    do {

        t++;
        i = P.at(j);

        Route route = Route(this->getProblem(), this->getConfig(), this->getDepot(), id);
        route.setRelaxDuration(false);
        id++;

        for (k = i; k < j; ++k) {
            //--trip.push_back(route.at(k));
            route.getTour().push_back(this->getGene().at(k));
        }

        route.calculateCost();
        this->getRoutes().push_back(route);

        // Depot
        //--trip.push_back(ROUTE_DELIM);
        j = i;

    } while (i > 0);

}

void Individual::splitExtractVehicleLimited(vector<int>& P) {

    int n = this->getGene().size();

    int p = 0, t = 0, j = n, i, k, id = 0;

    // Depot
    //--trip.push_back(ROUTE_DELIM);
    p++;

    // Repeat
    do {

        t++;
        i = P.at(j);

        // If the number of routes is less than the number of vehicles
        if (id < this->getProblem()->getVehicles()) {

            Route route = Route(this->getProblem(), this->getConfig(), this->getDepot(), id);
            id++;

            for (k = i; k < j; ++k) {
                //--trip.push_back(route.at(k));
                route.getTour().push_back(this->getGene().at(k));
            }

            route.calculateCost();
            this->getRoutes().push_back(route);

        }
        else {
            // Put in the last route
            for (k = i; k < j; ++k)
                this->getRoutes().at(this->getProblem()->getVehicles() - 1).addAtBack(this->getGene().at(k));
        }

        // Depot
        //--trip.push_back(ROUTE_DELIM);
        j = i;

    } while (i > 0);
}

void Individual::updateRoutesID() {

    int id = 0;
    for (auto ite = this->getRoutes().begin(); ite != this->getRoutes().end(); ++ite) {
        (*ite).setId(id);
        id++;
    }

}
