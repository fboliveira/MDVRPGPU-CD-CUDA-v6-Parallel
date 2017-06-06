/* 
 * File:   Route.cpp
 * Author: fernando
 * 
 * Created on July 21, 2014, 10:09 PM
 */

#include "Route.hpp"

/*
 * Constructors and Destructor
 */

Route::Route(MDVRPProblem *problem, AlgorithmConfig *config, int depot, int routeID) {

    this->setProblem(problem);
    this->setConfig(config);
    this->setDepot(depot);
    this->setId(routeID);
    this->startValues();
    this->setTour(list<int>());
    this->setRelaxDuration(false);

}

Route::Route(const Route& other) :
tour(other.tour), cost(other.cost), penaltyDuration(other.penaltyDuration),
penaltyDemand(other.penaltyDemand), id(other.id), depot(other.depot),
demand(other.demand), serviceTime(other.serviceTime),
relaxDuration(other.relaxDuration),
problem(other.problem), config(other.config) {
}

//Route::Route(const Route& route) {
//
//    try {
//
//        this->setProblem(route.getProblem());
//        this->setConfig(route.getConfig());
//
//        this->setId(route.getId());
//        this->setDepot(route.getDepot());
//        this->setDemand(route.getDemand());
//
//        this->setTour(list<int>());
//
//        if (!route.getTourConst().empty()) 
//            this->setTour(route.getTourConst());
//        //else
//        //    cout << "Route::copy => empty\n\n";
//
//        this->setCost(route.getCost());
//        this->setPenaltyDuration(route.getPenaltyDuration());
//        this->setPenaltyDemand(route.getPenaltyDemand());
//
//        if (this->getDepot() < 0)
//            cout << "Route::Route() => Dep invalido!\n";
//
//    } catch (exception &e) {
//        printf("%s\n", e.what());
//    }
//}

/*
 * Getters and Setters
 */

float Route::getCost() {

    if (this->cost < 0) {
        cout << "Error! Cost < 0" << endl;
        this->printSolution();
    }

    return this->cost;
}

void Route::setCost(float cost) {
    this->cost = cost;
    this->updatePenalty();
}

int Route::getServiceTime() const {
    return serviceTime;
}

void Route::setServiceTime(int serviceTime) {
    this->serviceTime = serviceTime;
}

bool Route::isRelaxDuration() const {
    return relaxDuration;
}

void Route::setRelaxDuration(bool relaxDuration) {
    this->relaxDuration = relaxDuration;
}

float Route::getPenaltyDuration() const {
    return this->penaltyDuration;
}

void Route::setPenaltyDuration(float penalty) {
    this->penaltyDuration = penalty;
}

int Route::getDemand() const {
    return this->demand;
}

void Route::setDemand(int demand) {
    this->demand = demand;
}

float Route::getPenaltyDemand() const {
    return this->penaltyDemand;
}

void Route::setPenaltyDemand(float penaltyDemand) {
    this->penaltyDemand = penaltyDemand;
}

int Route::getId() const {
    return this->id;
}

void Route::setId(int id) {
    this->id = id;
}

int Route::getDepot() const {
    return this->depot;
}

void Route::setDepot(int depot) {
    this->depot = depot;
}

list<int> Route::getTourConst() const {
    return this->tour;
}

list<int>& Route::getTour() {
    return this->tour;
}

void Route::setTour(list<int> tour) {
    this->tour = tour;
}

AlgorithmConfig* Route::getConfig() const {
    return this->config;
}

void Route::setConfig(AlgorithmConfig *config) {
    this->config = config;
}

MDVRPProblem* Route::getProblem() const {
    return this->problem;
}

void Route::setProblem(MDVRPProblem *problem) {
    this->problem = problem;
}

/* 
 * Methods
 */

void Route::setCustomersPosition(vector<CustomerPosition>& position) {

    for (auto i = this->getTour().begin(); i != this->getTour().end(); ++i) {
        int customer = (*i);

        position.at(customer - 1).setCustomer(customer);
        position.at(customer - 1).setDepot(this->getDepot());
        position.at(customer - 1).setRoute(this->getId());

    }

}

void Route::startValues() {
    this->setPenaltyDuration(0.0);
    this->setPenaltyDemand(0.0);
    this->setDemand(0);
    this->setServiceTime(0);
    this->setCost(0.0);
}

float Route::getTotalCost() {
    float totalCost = this->getCost() + this->getPenaltyDuration() + this->getPenaltyDemand();
    return totalCost;
}

void Route::updatePenalty() {

    if (this->getDemand() > this->getProblem()->getCapacity())
        this->setPenaltyDemand(this->getConfig()->getCapacityPenalty() * (this->getDemand() - this->getProblem()->getCapacity()));
    else
        this->setPenaltyDemand(0.0);
    
    float cost = this->getCost() + this->getServiceTime();
    
    if (this->getProblem()->getDuration() > 0 && cost > this->getProblem()->getDuration())
        this->setPenaltyDuration(this->getConfig()->getRouteDurationPenalty() * (cost - this->getProblem()->getDuration()));
    else
        this->setPenaltyDuration(0.0);       
}

// Add customer at the front of the list

typedef_listIntIterator Route::addAtFront(int customer) {

    float distFirstDep = 0.0, distFirstNew = 0.0, distNewDep = 0.0;

    // If the list is empty
    if (this->getTour().empty()) {
        return this->addAtBack(customer);
    } else {

        // Get the first customer        
        int firstCustomer = this->getTour().front();

        // Calculate the distance from the first to the depot
        distFirstDep = this->getProblem()->getDepotDistances().at(this->getDepot()).at(firstCustomer - 1);
        // Calculate the distance from the first to the new customer
        distFirstNew = this->getProblem()->getCustomerDistances().at(firstCustomer - 1).at(customer - 1);
        // Calculate the distance from the new to the depot
        distNewDep = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);

    }

    // Update demand
    this->setDemand(this->getDemand() + this->getProblem()->getDemand().at(customer - 1));
    // Update service
    this->setServiceTime(this->getServiceTime() + this->getProblem()->getServiceTime().at(customer - 1));

    this->setCost(this->getCost() - distFirstDep + distFirstNew + distNewDep);

    this->getTour().push_front(customer);

    return this->getTour().begin();

}

// Add customer at the end of the list

typedef_listIntIterator Route::addAtBack(int customer) {

    float distLastDep = 0.0, distLastNew = 0.0, distNewDep = 0.0;

    // If the list is empty
    if (this->getTour().empty()) {
        // D -> C -> D
        distNewDep = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1) * 2;
    } else {

        // Get the last customer        
        int lastCustomer = this->getTour().back();

        // Calculate the distance from the last to depot
        distLastDep = this->getProblem()->getDepotDistances().at(this->getDepot()).at(lastCustomer - 1);
        // Calculate the distance from the last to the new customer
        distLastNew = this->getProblem()->getCustomerDistances().at(lastCustomer - 1).at(customer - 1);
        // Calculate the distance from the new to depot
        distNewDep = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);

    }

    // Update demand
    this->setDemand(this->getDemand() + this->getProblem()->getDemand().at(customer - 1));
    // Update service
    this->setServiceTime(this->getServiceTime() + this->getProblem()->getServiceTime().at(customer - 1));
    this->setCost(this->getCost() - distLastDep + distLastNew + distNewDep);

    this->getTour().push_back(customer);
    return prev(this->getTour().end());

}

// Add customer after previous one

//template<typename Iter>
//void Route::addAfterPrevious(Iter previous, int customer) {

typedef_listIntIterator Route::addAfterPrevious(typedef_listIntIterator previous, int customer) {

    try {

        if (this->getTour().empty())
            return this->addAtBack(customer);
        else if (previous == this->getTour().end() || next(previous) == this->getTour().end()) {
            return this->addAtBack(customer);
        } else {

            int prevCustomer = *previous;

            auto nextPosition = previous;
            nextPosition++;
            int nextCustomer = *nextPosition;

            float distPrevAfter = this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(nextCustomer - 1);
            float distPrevNew = this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            float distNewAfter = this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);

            // Update demand    
            this->setDemand(this->getDemand() + this->getProblem()->getDemand().at(customer - 1));
            // Update service
            this->setServiceTime(this->getServiceTime() + this->getProblem()->getServiceTime().at(customer - 1));

            this->setCost(this->getCost() - distPrevAfter + distPrevNew + distNewAfter);
            return this->getTour().insert(nextPosition, customer);

        }

    } catch (exception &e) {
        cout << endl;
        cout << endl;
        this->printSolution();
        cout << "Route::addAfterPrevious: " << e.what() << '\n';
    }

    return this->find(customer);

}

typedef_listIntIterator Route::addAfterPrevious(int previousCustomer, int customer) {

    if (previousCustomer <= 0)
        return this->addAtFront(customer);

    auto previous = this->find(previousCustomer);
    return this->addAfterPrevious(previous, customer);

}

void Route::insertBestPosition(int customer) {

    float bestCost;
    bool front = true;

    //cout << this->getTour().size() << endl;

    typedef_listIntIterator bestPos;
    typedef_listIntIterator pos = this->addAtFront(customer);

    bestCost = this->getTotalCost();

    this->remove(pos);

    for (auto ite = this->getTour().begin(); ite != this->getTour().end(); ++ite) {

        pos = this->addAfterPrevious(ite, customer);

        if (Util::isBetterSolution(this->getTotalCost(), bestCost)) {
            bestCost = this->getTotalCost();
            bestPos = ite;
            front = false;
        }

        this->remove(pos);

    }

    if (front)
        this->addAtFront(customer);
    else
        this->addAfterPrevious(bestPos, customer);

}

typedef_listIntIterator Route::find(int customer) {
    return std::find(this->getTour().begin(), this->getTour().end(), customer);
}

void Route::remove(typedef_listIntIterator position) {

    float previous, after, newCost, cost = 0;

    int customer = *position;

    // Just one node
    if (customer == this->getTour().front() && customer == this->getTour().back()) {
        this->getTour().clear();
        this->startValues();
    } else {

        if (customer == this->getTour().front()) {

            auto nextPosition = position;
            nextPosition++;
            int nextCustomer = *nextPosition;

            // From Depot to Customer
            previous = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);
            // From Customer to Next
            after = this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);
            // From Next to Depot
            newCost = this->getProblem()->getDepotDistances().at(this->getDepot()).at(nextCustomer - 1);

        } else if (customer == this->getTour().back()) {

            auto prevPosition = position;
            prevPosition--;
            int prevCustomer = *prevPosition;

            previous = this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            after = this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);
            newCost = this->getProblem()->getDepotDistances().at(this->getDepot()).at(prevCustomer - 1);

        } else {

            auto nextPosition = position;
            nextPosition++;
            int nextCustomer = *nextPosition;

            auto prevPosition = position;
            prevPosition--;
            int prevCustomer = *prevPosition;

            previous = this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            after = this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);
            newCost = this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(nextCustomer - 1);

        }

        this->setDemand(this->getDemand() - this->getProblem()->getDemand().at(customer - 1));
        // Update service
        this->setServiceTime(this->getServiceTime() - this->getProblem()->getServiceTime().at(customer - 1));

        cost = this->getCost() - previous - after + newCost;
        this->setCost(this->getCost() - previous - after + newCost);

        this->getTour().erase(position);

    }

    //this->calculateCost();

    if ( fabsf(cost - Util::scaledFloat(this->getCost())) > 0.1) {
        this->printSolution();
        cout << cost << endl;
    }

}

void Route::remove(int customer) {


    this->remove(this->find(customer));
    //auto position = this->find(customer); //find(this->getTour().begin(), this->getTour().end(), customer);
    //this->remove(position);

}

void Route::calculateCost() {

    int demand = 0;
    float cost = 0.0;
    int service = 0;

    if (!this->getTour().empty()) {

        int customer = this->getTour().front();

        if (customer > this->getProblem()->getCustomers())
            cout << "Erro => " << customer << "\n";

        // D->C1
        cost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);
        // Cn->D
        customer = this->getTour().back();
        cost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);

        int nextCustomer;

        for (auto i = this->getTour().begin(); i != this->getTour().end(); ++i) {

            customer = *i;

            auto nextPosition = next(i);

            if (nextPosition != this->getTour().end()) {

                nextCustomer = *nextPosition;
                cost += this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);
            }

            demand += this->getProblem()->getDemand().at(customer - 1);
            service += this->getProblem()->getServiceTime().at(customer - 1);

        }

    }

    this->setDemand(demand);
    this->setServiceTime(service);
    this->setCost(cost);

}

///// return  d[s[i_antes]][s[i]] + d[s[i]][s[i_depois]] + d[s[j_antes]][s[j]] + d[s[j]][s[j_depois]];

float Route::calculateCost(typedef_listIntIterator start, typedef_listIntIterator end, int& demand) {

    float before, after, cost = 0;
    int service = 0;

    demand = 0;

    for (auto position = start; position != end; ++position) {

        if (position == this->getTour().begin()) {
            //cout << "D ";
            before = this->getProblem()->getDepotDistances().at(this->getDepot()).at((*position) - 1);
        } else {
            before = this->getProblem()->getCustomerDistances().at((*prev(position)) - 1).at((*position) - 1);
            //cout << *prev(position);
        }

        //cout << " <-> " << (*position) << " <-> ";

        if (next(position) == this->getTour().end()) {
            after = this->getProblem()->getDepotDistances().at(this->getDepot()).at((*position) - 1);
            //cout << " D";
        } else {

            after = this->getProblem()->getCustomerDistances().at((*position) - 1).at((*next(position)) - 1);
            //cout << (*next(position));
        }

        demand += this->getProblem()->getDemand().at((*position) - 1);
        //service += this->getProblem()->getServiceTime().at((*position) - 1);

        cost += before + after;
        //cout << endl;
    }

    return cost;

}

void Route::changeCustomer(typedef_listIntIterator position, int newCustomer) {

    float oldCustomerCost = 0, newCustomerCost = 0;

    int customer = (*position);

    // Just one node
    if (customer == this->getTour().front() && customer == this->getTour().back()) {
        this->remove(position);
        this->addAtBack(newCustomer);
    } else {

        auto nextPosition = next(position);

        // If it is in the front of
        if (position == this->getTour().begin()) {

            int nextCustomer = (*nextPosition);

            // D->C
            oldCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);
            // C->C+1
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);

            // D->NewC
            newCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(newCustomer - 1);
            // NewC->C+1
            newCustomerCost += this->getProblem()->getCustomerDistances().at(newCustomer - 1).at(nextCustomer - 1);

        } else if (next(position) == this->getTour().end()) { // Last one

            auto prevPosition = prev(position);
            int prevCustomer = (*prevPosition);

            // C-1->C
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            // C->D
            oldCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);

            // C-1->NewC
            newCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(newCustomer - 1);

            // D->NewC
            newCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(newCustomer - 1);

        } else { // Anywhere...

            auto prevPosition = prev(position);
            int nextCustomer = (*nextPosition);
            int prevCustomer = (*prevPosition);

            // C-1->C
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            // C->C+1
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);

            // C-1->NewC
            newCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(newCustomer - 1);
            // NewC->C+1
            newCustomerCost += this->getProblem()->getCustomerDistances().at(newCustomer - 1).at(nextCustomer - 1);

        }

        this->setDemand(this->getDemand() - this->getProblem()->getDemand().at(customer - 1)
                + this->getProblem()->getDemand().at(newCustomer - 1));

        // Update service
        this->setServiceTime(this->getServiceTime() - this->getProblem()->getServiceTime().at(customer - 1)
                + this->getProblem()->getServiceTime().at(newCustomer - 1));

        this->setCost(this->getCost() - oldCustomerCost + newCustomerCost);

        // Change value of customer
        *position = newCustomer;

    }

}

void Route::swap(typedef_listIntIterator source, typedef_listIntIterator dest) {

    int customerSource = (*source);
    int customerDest = (*dest);

    this->changeCustomer(source, customerDest);
    this->changeCustomer(dest, customerSource);

}

void Route::reverse(typedef_listIntIterator begin, typedef_listIntIterator end) {

    // Invert from BEGIN to END
    while (begin != end) {

        // A-> End
        // B-> Start
        this->swap(begin, end);

        begin++;
        end--;

        if (begin == next(end) || end == prev(begin))
            break;
    }

}

bool Route::isPenalized() {

    return this->getPenaltyDemand() > 0 || this->getPenaltyDuration() > 0;
}

float Route::getDuration() const {
    return this->getProblem()->getDurationConditional(this->isRelaxDuration());
}

int Route::getCapacity() const {
    return this->getProblem()->getCapacityConditional(this->isRelaxDuration());
}

void Route::routeToVector(int* route) {

    int idx = 0;

    for (auto i = this->getTour().begin(); i != this->getTour().end(); ++i) {
        route[idx] = (*i);
        idx++;
    }

}

void Route::vectorToRoute(int* route, int size) {
    this->vectorToRoute(route, 0, size - 1);
}

void Route::vectorToRoute(int* route, int first, int last) {

    this->getTour().clear();
    this->startValues();

    for (int i = first; i <= last; ++i)
        if (route[i] > 0)
            this->addAtBack(route[i]);

}

void Route::print() {

    if (this->getTour().empty())
        return;

    cout << "[D: " << this->getDepot() << " - R: " << this->getId() << "] => ";
    cout << "Cost: " << this->getCost()
            << " + Service: " << this->getServiceTime()
            << " + P_Dur: " << this->getPenaltyDuration()
            << " + P_Dem: " << this->getPenaltyDemand()
            << " => TOTAL = " << this->getTotalCost() + this->getServiceTime() << endl;

    cout << "Demand: " << this->getDemand() << " - Srv: " << this->getServiceTime() << " => Route: D -> ";

    int num = 0;
    MDVRPProblem * problem = this->getProblem();

    for_each(this->getTour().begin(), this->getTour().end(), [&num, problem] (int customer) {

        cout << customer << " (" << problem->getDemand().at(customer - 1)
                << " - Srv: " << problem->getServiceTime().at(customer - 1)
                << ") -> ";
        num++;
    });

    cout << "D [ " << num << " ]\n\n";

}

void Route::printSolution() {

    if (this->getTour().empty())
        return;

    cout << this->getDepot() + 1 << "\t" << this->getId() + 1 << "\t";
    printf("%.2f\t%d\t%d", this->getTotalCost(), this->getServiceTime(), this->getDemand());

    cout << "\t0 ";

    int num = 0;
    MDVRPProblem * problem = this->getProblem();

    for_each(this->getTour().begin(), this->getTour().end(), [&num, problem] (int customer) {

        cout << customer << " ";
        num++;
    });

    cout << "0\n";
}

bool Route::operator==(const Route& right) const {
    bool result = this->getTourConst() == right.getTourConst(); // Compare right and *this here
    return result;
}

/*
 * Static functions
 */


