/*
* File:   MDVRPProblem.cpp
* Author: fernando
*
* Created on July 21, 2014, 2:10 PM
*/

#include "MDVRPProblem.hpp"

/*
* Constructors
*/

MDVRPProblem::MDVRPProblem() {
    //createStreams();
}

MDVRPProblem::~MDVRPProblem(){
    //destroyStreams();
}

/*
* Getters and Setters
*/

typedef_vectorMatrix<int>& MDVRPProblem::getAllocation() {
    return this->allocation;
}

float MDVRPProblem::getBestKnowSolution() const {
    return this->bestKnowSolution;
}

void MDVRPProblem::setBestKnowSolution(float bestKnowSolution) {
    this->bestKnowSolution = bestKnowSolution;
}

int MDVRPProblem::getCapacity() const {
    return this->capacity;
}

void MDVRPProblem::setCapacity(int capacity) {
    this->capacity = capacity;
}

// Distance [CLI][CLI];

typedef_vectorMatrix<float>& MDVRPProblem::getCustomerDistances() {
    return this->customerDistances;
}

vector<typedef_point>& MDVRPProblem::getCustomerPoints() {
    return this->customerPoints;
}

int MDVRPProblem::getCustomers() const {
    return this->customers;
}

void MDVRPProblem::setCustomers(int customers) {
    this->customers = customers;
}

vector<int>& MDVRPProblem::getDemand() {
    return this->demand;
}

vector<int>& MDVRPProblem::getServiceTime() {
    return serviceTime;
}

int MDVRPProblem::getDepots() const {
    return this->depots;
}

void MDVRPProblem::setDepots(int depot) {
    this->depots = depot;
}

// Distance [DEP][CLI];

typedef_vectorMatrix<float>& MDVRPProblem::getDepotDistances() {
    return this->depotDistances;
}

vector<typedef_point>& MDVRPProblem::getDepotPoints() {
    return this->depotPoints;
}

float MDVRPProblem::getDuration() const {
    return this->duration;
}

void MDVRPProblem::setDuration(float duration) {
    this->duration = duration;
}

float MDVRPProblem::getDurationConditional(bool relaxed) const {

    float routeDuration;

    if (false)
        routeDuration = 2 * this->getDuration();
    else
        routeDuration = this->getDuration();

    return routeDuration;

}

int MDVRPProblem::getCapacityConditional(bool relaxed) const {

    int capacity;

    if (relaxed)
        capacity = 2 * this->getCapacity();
    else
        capacity = this->getCapacity();

    return capacity;

}

std::string MDVRPProblem::getInstCode() const {
    return this->instCode;
}

void MDVRPProblem::setInstCode(std::string instCode) {
    std::transform(instCode.begin(), instCode.end(), instCode.begin(), ::toupper);
    this->instCode = instCode;
}

std::string MDVRPProblem::getInstance() const {
    return this->instance;
}

void MDVRPProblem::setInstance(std::string instance) {
    this->instance = instance;
}

typedef_vectorMatrix<int>& MDVRPProblem::getNearestCustomerFromCustomer() {
    return this->nearestCustomerFromCustomer;
}

typedef_vectorMatrix<int>& MDVRPProblem::getNearestCustomersFromDepot() {
    return this->nearestCustomersFromDepot;
}

typedef_vectorMatrix<int>& MDVRPProblem::getNearestDepotsFromCustomer() {
    return this->nearestDepotsFromCustomer;
}

int MDVRPProblem::getVehicles() const {
    return this->vehicles;
}

void MDVRPProblem::setVehicles(int vehicles) {
    this->vehicles = vehicles;
}

void MDVRPProblem::setAllocation(typedef_vectorMatrix<int> allocation) {
    this->allocation = allocation;
}

void MDVRPProblem::setCustomerDistances(typedef_vectorMatrix<float> customerDistances) {
    this->customerDistances = customerDistances;
}

void MDVRPProblem::setCustomerPoints(vector<typedef_point> customerPoints) {
    this->customerPoints = customerPoints;
}

void MDVRPProblem::setDemand(vector<int> demand) {
    this->demand = demand;
}

void MDVRPProblem::setServiceTime(vector<int> serviceTime) {
    this->serviceTime = serviceTime;
}

void MDVRPProblem::setDepotDistances(typedef_vectorMatrix<float> depotDistances) {
    this->depotDistances = depotDistances;
}

void MDVRPProblem::setDepotPoints(vector<typedef_point> depotPoints) {
    this->depotPoints = depotPoints;
}

void MDVRPProblem::setNearestCustomerFromCustomer(typedef_vectorMatrix<int> nearestCustomerFromCustomer) {
    this->nearestCustomerFromCustomer = nearestCustomerFromCustomer;
}

void MDVRPProblem::setNearestCustomersFromDepot(typedef_vectorMatrix<int> nearestCustomersFromDepot) {
    this->nearestCustomersFromDepot = nearestCustomersFromDepot;
}

void MDVRPProblem::setNearestDepotsFromCustomer(typedef_vectorMatrix<int> nearestDepotsFromCustomer) {
    this->nearestDepotsFromCustomer = nearestDepotsFromCustomer;
}

double MDVRPProblem::getAvgCustomerDistance() const {
    return avgCustomerDistance;
}

void MDVRPProblem::setAvgCustomerDistance(double avgCustomerDistance) {
    this->avgCustomerDistance = avgCustomerDistance;
}

double MDVRPProblem::getAvgDepotDistance() const {
    return avgDepotDistance;
}

void MDVRPProblem::setAvgDepotDistance(double avgDepotDistance) {
    this->avgDepotDistance = avgDepotDistance;
}

typedef_vectorMatrix<int>& MDVRPProblem::getGranularNeighborhood() {
    return granularNeighborhood;
}

void MDVRPProblem::setGranularNeighborhood(typedef_vectorMatrix<int> granularNeighborhood) {
    this->granularNeighborhood = granularNeighborhood;
}

ManagedMatrix<float>& MDVRPProblem::getMngCustomerDistances() {
    return mngCustomerDistances;
}

ManagedMatrix<float>& MDVRPProblem::getMngDepotDistances(){
    return mngDepotDistances;
}

ManagedArray<int>& MDVRPProblem::getMngDemand() {
    return mngDemand;
}

ManagedArray<int>& MDVRPProblem::getMngServiceTime() {
    return mngServiceTime;
}

/*
* Method
*/

void MDVRPProblem::allocateMemory() {

    this->setCustomerPoints(vector<typedef_point>(this->getCustomers()));
    this->setDepotPoints(vector<typedef_point>(this->getDepots()));

    this->setDemand(vector<int>(this->getCustomers()));
    this->setServiceTime(vector<int>(this->getCustomers()));

    typedef_vectorMatrix<float> customerDistances = typedef_vectorMatrix<float>(this->getCustomers());
    for (int i = 0; i < this->getCustomers(); ++i)
        customerDistances.at(i).resize(this->getCustomers());

    this->setCustomerDistances(customerDistances);

    typedef_vectorMatrix<float> depotDistances = typedef_vectorMatrix<float>(this->getDepots());

    for (int i = 0; i < this->getDepots(); ++i)
        depotDistances.at(i).resize(this->getCustomers());

    this->setDepotDistances(depotDistances);

    typedef_vectorMatrix<int> nearestCustomerFromCustomer = typedef_vectorMatrix<int>(this->getCustomers()); //  allocate_matrix_int(clientes, clientes);

    for (int i = 0; i < this->getCustomers(); ++i)
        nearestCustomerFromCustomer.at(i).resize(this->getCustomers());
    this->setNearestCustomerFromCustomer(nearestCustomerFromCustomer);

    typedef_vectorMatrix<int> nearestDepotsFromCustomer = typedef_vectorMatrix<int>(this->getCustomers()); //   allocate_matrix_int(clientes, depositos);

    for (int i = 0; i < this->getCustomers(); ++i)
        nearestDepotsFromCustomer.at(i).resize(this->getDepots());
    this->setNearestDepotsFromCustomer(nearestDepotsFromCustomer);

    typedef_vectorMatrix<int> nearestCustomersFromDepot = typedef_vectorMatrix<int>(this->getDepots()); //   allocate_matrix_int(depositos, clientes);

    for (int i = 0; i < this->getDepots(); ++i)
        nearestCustomersFromDepot.at(i).resize(this->getCustomers());

    this->setNearestCustomersFromDepot(nearestCustomersFromDepot);

    typedef_vectorMatrix<int> allocation = typedef_vectorMatrix<int>(this->getCustomers()); // allocate_matrix_int(clientes, depositos);

    for (int i = 0; i < this->getCustomers(); ++i)
        allocation.at(i).resize(this->getDepots());

    this->setAllocation(allocation);

    // Granular Neighborhoods
    typedef_vectorMatrix<int> granularNeighborhood = typedef_vectorMatrix<int>(this->getCustomers()); // allocate_matrix_int(clientes, depositos);

    for (int i = 0; i < this->getCustomers(); ++i)
        granularNeighborhood.at(i).resize(this->getCustomers());

    this->setGranularNeighborhood(granularNeighborhood);

    // GPU allocation
    this->getMngCustomerDistances().setLines(this->getCustomers());
    this->getMngCustomerDistances().setColumns(this->getCustomers());
    this->getMngCustomerDistances().init();

    this->getMngDepotDistances().setLines(this->getDepots());
    this->getMngDepotDistances().setColumns(this->getCustomers());
    this->getMngDepotDistances().init();

    this->getMngDemand().setCols(this->getCustomers());
    this->getMngDemand().init();

    this->getMngServiceTime().setCols(this->getCustomers());
    this->getMngServiceTime().init();

}

bool MDVRPProblem::processInstanceFiles(char* dataFile, char* solutionFile, char* instCode) {

    FILE *data;
    FILE *solution;
    int i, type, vehicles, customers, depots, valueInt;
    float valueFloat;
    typedef_point point;
    //char null;

    data = fopen(dataFile, "r");
    solution = fopen(solutionFile, "r");

    if (data == NULL) {
        printf("Erro ao abrir o arquivo = %s \n\n", dataFile);
        return false;
    }

    if (solution == NULL) {
        printf("Erro ao abrir o arquivo = %s \n\n", dataFile);
        return false;
    }

    this->setInstCode(instCode);

    // Problem informations
    fscanf(data, "%d %d %d %d", &type, &vehicles, &customers, &depots);
    this->setDepots(depots);
    this->setVehicles(vehicles);
    this->setCustomers(customers);

    // Allocate memory using problem informations
    this->allocateMemory();

    // Instance
    this->setInstance(dataFile);

    // Best know solution
    fscanf(solution, "%f", &valueFloat);
    this->setBestKnowSolution(valueFloat);

    // Capacity and duration - homogeneous problems    
    for (i = 0; i < this->getDepots(); ++i) {
        //fscanf(data, "%f %d", &problema.duracao, &problema.capacidade);
        fscanf(data, "%f %d", &valueFloat, &valueInt);
        this->setDuration(valueFloat);
        this->setCapacity(valueInt);
    }

    // Customer informations
    int id, service, demand;

    for (i = 0; i < this->getCustomers(); ++i) {
        // Id, X, Y, service duration, demand
        fscanf(data, "%d %f %f %d %d", &id, &point.x, &point.y, &service, &demand);

        this->getCustomerPoints().at(i) = point;

        this->getServiceTime().at(i) = service;
        this->getDemand().at(i) = demand;

        this->getMngDemand().set(i, demand);
        this->getMngServiceTime().set(i, service);

        while (getc(data) != '\n');
    }

    // Depot informations
    for (i = 0; i < this->getDepots(); ++i) {
        fscanf(data, "%d %f %f", &type, &point.x, &point.y);
        this->getDepotPoints().at(i) = point;

        while (getc(data) != '\n' && !feof(data));
    }

    this->calculateMatrixDistance();
    this->setNearestCustomersFromCustomer();
    this->setNearestCustomersFromDepot();
    this->setNearestDepotsFromCustomer();

    this->defineIntialCustomersAllocation();

    this->operateGranularNeighborhood();

    fclose(solution);
    fclose(data);

    return true;

}

void MDVRPProblem::calculateMatrixDistance() {

    int i, j;
    float dist;
    double totalDistance = 0.0;

    for (i = 0; i < this->getCustomers(); ++i) {
        for (j = 0; j < this->getCustomers(); ++j) {

            if (i != j) {

                dist = Util::calculateEucDist2D(this->getCustomerPoints().at(i).x,
                    this->getCustomerPoints().at(i).y,
                    this->getCustomerPoints().at(j).x,
                    this->getCustomerPoints().at(j).y);

                this->getCustomerDistances().at(i).at(j) = dist;
                this->getMngCustomerDistances().set(i, j, dist);
                totalDistance += dist;
            }
            else {
                // Seta distancia para I == J para INT_MAX para recuperar o
                // menor
                this->getCustomerDistances().at(i).at(j) = powf(10, 3);
            }
        }
    }

    // n - 1 Customers
    totalDistance /= (double)(this->getCustomers() * this->getCustomers());
    this->setAvgCustomerDistance(totalDistance);
    totalDistance = 0.0;

    // Calcula a distancia dos clientes para o deposito
    for (i = 0; i < this->getDepots(); ++i) {

        for (j = 0; j < this->getCustomers(); ++j) {

            dist = Util::calculateEucDist2D(this->getDepotPoints().at(i).x,
                this->getDepotPoints().at(i).y,
                this->getCustomerPoints().at(j).x,
                this->getCustomerPoints().at(j).y);

            this->getDepotDistances().at(i).at(j) = dist;
            this->getMngDepotDistances().set(i, j, dist);
            totalDistance += dist;

        }
    }

    // Relation from customers to depots
    totalDistance /= (double)(this->getCustomers() * 1.0);
    this->setAvgDepotDistance(totalDistance);

}

void MDVRPProblem::setNearestCustomersFromCustomer() {

    int i, j;
    vector<typedef_order> order = vector<typedef_order>(this->getCustomers());

    for (i = 0; i < this->getCustomers(); ++i) {

        // Get distances
        for (j = 0; j < this->getCustomers(); ++j) {
            typedef_order od;
            od.index = j;
            od.cost = this->getCustomerDistances().at(i).at(j);
            order.at(j) = od;
        }

        // Order
        Util::selectVectorOrder(order);

        // Put back in nearest order
        for (j = 0; j < this->getCustomers(); ++j) {
            this->getNearestCustomerFromCustomer().at(i).at(j) = order.at(j).index;
        }
    }

}

void MDVRPProblem::setNearestDepotsFromCustomer() {

    int i, j;
    vector<typedef_order> order = vector<typedef_order>(this->getDepots());


    for (i = 0; i < this->getCustomers(); ++i) {

        // Get distances
        for (j = 0; j < this->getDepots(); ++j) {
            typedef_order od;
            od.index = j;
            od.cost = this->getDepotDistances().at(j).at(i); // DxC
            order.at(j) = od;
        }

        // Order
        Util::selectVectorOrder(order);

        // Put back in nearest order
        for (j = 0; j < this->getDepots(); ++j) {
            this->getNearestDepotsFromCustomer().at(i).at(j) = order.at(j).index;
        }
    }

}

void MDVRPProblem::setNearestCustomersFromDepot() {

    int i, j;
    vector<typedef_order> order = vector<typedef_order>(this->getCustomers());

    for (i = 0; i < this->getDepots(); ++i) {

        // Get distances
        for (j = 0; j < this->getCustomers(); ++j) {
            typedef_order od;
            od.index = j;
            od.cost = this->getDepotDistances().at(i).at(j); // DxC
            order.at(j) = od;
        }

        // Order
        Util::selectVectorOrder(order);

        // Put back in nearest order
        for (j = 0; j < this->getCustomers(); ++j) {
            this->getNearestCustomersFromDepot().at(i).at(j) = order.at(j).index;
        }
    }

}

void MDVRPProblem::getDepotGroup(int depot, vector<int>& customers) {

    int customer;

    for (int c = 0; c < this->getCustomers(); ++c) {

        // Zero reference
        customer = this->getNearestCustomersFromDepot()[depot][c];

        if (this->getAllocation().at(customer).at(depot) > 0)
            customers.push_back(customer);
    }

}

void MDVRPProblem::getNearestCustomerFromCustomerOnDepot(int customer, int depot, vector<int>& customers) {

    int c, near;

    for (c = 0; c < this->getCustomers(); ++c) {

        near = this->getNearestCustomerFromCustomer()[customer][c];

        if (this->getAllocation().at(near).at(depot) > 0)
            customers.push_back(near);
    }

}

void MDVRPProblem::defineIntialCustomersAllocation() {

    for (int c = 0; c < this->getCustomers(); ++c) {
        setCustomerOnDepot(c);
    }

}

void MDVRPProblem::setCustomerOnDepot(int customer) {

    int d;

    // Get the minimal distance from depots
    d = this->getNearestDepotsFromCustomer()[customer][0];
    float min = this->getDepotDistances()[d][customer];

    // Distance max with delta = 20%
    min = min * 1.2f;

    // Get the depot(s) with this distance and set
    for (d = 0; d < this->getDepots(); ++d)
        if (this->getDepotDistances()[d][customer] <= min)
            this->getAllocation()[customer][d] = 1;

    // Get the nearest customer
    int nearest = this->getNearestCustomerFromCustomer()[customer][0];
    int depNearest = this->getNearestDepotsFromCustomer()[nearest][0];

    this->getAllocation()[customer][depNearest] = 1;

}

Monitor& MDVRPProblem::getMonitor() {
    return monitor;
}

cudaStream_t MDVRPProblem::getStream(int id) {
    if (streams[id] == NULL)
        cudaStreamCreate(&streams[id]);

    return streams[id];
}

void MDVRPProblem::print() {

    int i, j;

    printf("Instancia = %s \n\n", this->getInstance().c_str());
    printf("Minimo = %.2f\n\n", this->getBestKnowSolution());

    printf("Depositos = %d\n", this->getDepots());
    printf("Veiculos = %d\n", this->getVehicles());
    printf("Clientes = %d\n", this->getCustomers());

    printf("\n\n");

    for (i = 0; i < this->getDepots(); ++i)
        printf("D: %i -> Pontos: (%.2f, %.2f) -- Capacidade: %d -- Duracao: %.2f\n", i + 1,
        this->getDepotPoints()[i].x, this->getDepotPoints()[i].y, this->getCapacity(), this->getDuration());

    printf("\n\n");

    for (i = 0; i < this->getCustomers(); ++i)
        printf("Cliente: %d - Pontos: (%.2f, %.2f) - Demanda: %d - Servico: %d\n", i + 1, this->getCustomerPoints()[i].x,
        this->getCustomerPoints()[i].y, this->getDemand()[i], this->getServiceTime().at(i));

    // Distancia
    printf("\n\nMatriz de distancias - DEPOSITOS x CLIENTES:\n\n");
    for (i = 0; i < this->getDepots(); ++i) {
        for (j = 0; j < this->getCustomers(); ++j) {
            printf("%.4f\t", this->getDepotDistances()[i][j]);
        }
        printf("\n");
    }

    // Distancia
    printf("\n\nMatriz de distancias - CLIENTES x CLIENTES:\n\n");
    for (i = 0; i < this->getCustomers(); ++i) {
        for (j = 0; j < this->getCustomers(); ++j) {
            printf("%.4f\t", this->getCustomerDistances()[i][j]);
        }
        printf("\n");
    }



    //
    //    printf("\n\n");
    //
    //    printf("NEAREST CUSTOMERS FROM CUSTOMER:\n");
    //    for (i = 0; i < this->getCustomers(); ++i) {
    //        printf("Customer: %d => ", i + 1);
    //        Util::print(this->getNearestCustomerFromCustomer()[i]);
    //    }
    //    
    //    printf("\n\n");
    //
    //    printf("NEAREST CUSTOMERS FROM DEPOT:\n");
    //    for (i = 0; i < this->getDepots(); ++i) {
    //        printf("Depot: %d => ", i + 1);
    //        Util::print(this->getNearestCustomersFromDepot()[i]);
    //    }
    //
    //    printf("\n\n");
    //
    //    printf("NEAREST DEPOTS FROM CUSTOMER:\n");
    //    for (i = 0; i < this->getCustomers(); ++i) {
    //        printf("Customer: %d => ", i + 1);
    //        Util::print(this->getNearestDepotsFromCustomer()[i]);
    //    }

    printf("\n\n");

}

void MDVRPProblem::printAllocation() {

    printf("\nGLOBAL MEMORY ALLOCATION:\n");
    for (int i = 0; i < this->getCustomers(); ++i) {
        printf("Customer : %d => ", i + 1);
        Util::print(this->getAllocation()[i]);
    }

}

void MDVRPProblem::printAllocationDependecy() {

    int dep = 0, count;

    for (int i = 0; i < this->getCustomers(); ++i) {
        //printf("Customer : %d => ", i + 1);
        //Util::print(this->getAllocation()[i]);
        count = 0;
        for (int d = 0; d < this->getDepots(); ++d) {
            if (this->getAllocation().at(i).at(d) > 0)
                count++;
        }

        if (count > 1)
            dep++;
    }

    printf("%s: %d/%d\n", this->getInstCode().c_str(), dep, this->getCustomers());

}

void MDVRPProblem::operateGranularNeighborhood() {

    // granularity threshold value θ = βz (where β is a sparsification factor and z is the average cost of the edges)
    float beta = 1.2f;

    // distance factorφij =2cij +δj(∀i ∈ I, j ∈ J)isnotgreaterthanthemaximumduration D.

    double threshold = beta * this->getAvgCustomerDistance();

    /*
    cout << endl << endl;
    cout << "Avg Customer Distance: " << this->getAvgCustomerDistance() << endl;
    cout << "Granularity threshold: " << threshold << endl;
    cout << endl << endl;
    */

    for (int i = 0; i < this->getCustomers(); ++i) {
        for (int j = 0; j < this->getCustomers(); ++j) {

            this->getGranularNeighborhood().at(i).at(j) = 1;

            if (i != j)
                if (2 * this->getCustomerDistances().at(i).at(j) <= threshold)
                    this->getGranularNeighborhood().at(i).at(j) = 1;
        }
    }

    /*
    for(int i = 0; i < this->getCustomers(); ++i) {
    for(int j = 0; j < this->getCustomers(); ++j) {
    cout << d_factor[i][j] << " ";
    }
    cout << endl;
    }
    */
}

void MDVRPProblem::createStreams() {
    for (int i = 0; i < stm; ++i)
        cudaStreamCreate(&streams[i]);
}

void MDVRPProblem::destroyStreams() {
    for (int i = 0; i < stm; ++i)
        cudaStreamDestroy(streams[i]);

    delete streams;
}