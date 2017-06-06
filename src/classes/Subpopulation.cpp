/* 
 * File:   Subpopulation.cpp
 * Author: fernando
 * 
 * Created on July 22, 2014, 4:38 PM
 */

#include "Subpopulation.hpp"

/*
 * Constructors and Destructor
 */

Subpopulation::Subpopulation() {

}

Subpopulation::Subpopulation(MDVRPProblem* problem, AlgorithmConfig* config, EliteGroup* eliteGroup, int depot) :
problem(problem), config(config), eliteGroup(eliteGroup), depot(depot) {
    this->createPairingStructure();
    this->setBest(Individual(this->getProblem(), this->getConfig(), this->getDepot(), -1));

}

Subpopulation::Subpopulation(const Subpopulation& other) :
problem(other.problem), config(other.config), eliteGroup(other.eliteGroup),
depot(other.depot), best(other.best), individuals(other.individuals),
pairing(other.pairing) {

}

Subpopulation::~Subpopulation() {
    //delete[] locks;
}

/*
 * Getters and Setters
 */

Individual& Subpopulation::getBest() {
    return this->best;
}

void Subpopulation::setBest(Individual best) {
    this->best = best;
}

int Subpopulation::getDepot() const {
    return this->depot;
}

void Subpopulation::setDepot(int depot) {
    this->depot = depot;
}

//int Subpopulation::getId() const {
//    return this->id;
//}
//
//void Subpopulation::setId(int id) {
//    this->id = id;
//}

IndividualsGroup& Subpopulation::getIndividualsGroup() {
    return this->individuals;
}

void Subpopulation::setIndividualsGroup(IndividualsGroup individuals) {
    this->individuals = individuals;
}

vector<Pairing>& Subpopulation::getPairing() {
    return this->pairing;
}

void Subpopulation::setPairing(vector<Pairing> solutions) {
    this->pairing = solutions;
}

AlgorithmConfig* Subpopulation::getConfig() const {
    return this->config;
}

void Subpopulation::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* Subpopulation::getProblem() const {
    return this->problem;
}

void Subpopulation::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

EliteGroup* Subpopulation::getEliteGroup() {
    return this->eliteGroup;
}

void Subpopulation::setEliteGroup(EliteGroup* eliteGroup) {
    this->eliteGroup = eliteGroup;
}

bool Subpopulation::isLocked() const {
    return locked;
}

void Subpopulation::setLocked(bool locked) {

    //    if (locked)
    //        this->getProblem()->getMonitor().getMutexLocker().lock();
    //    else
    //        this->getProblem()->getMonitor().getMutexLocker().unlock();

    this->locked = locked;
}

/*
 * Public Methods
 */

void Subpopulation::createIndividuals() {

    this->getIndividualsGroup().clear();

    this->getIndividualsGroup().setProblem(this->getProblem());
    this->getIndividualsGroup().setConfig(this->getConfig());
    this->getIndividualsGroup().setDepot(this->getDepot());

    //cout << "\n\nSubpop: " << this->getDepot() << endl;

    for (int ind = 0; ind < this->getConfig()->getNumSubIndDepots(); ++ind) {
        Individual individual = Individual(this->getProblem(), this->getConfig(), this->getDepot(), ind);
        individual.create();
        //individual.print(true);
        this->getIndividualsGroup().add(individual);
    }

    //this->createLockers();

}

void Subpopulation::createPairingStructure() {

    this->getPairing().clear();

    for (int id = 0; id < this->getConfig()->getNumSubIndDepots(); ++id) {
        Pairing pairing = Pairing(this->getProblem(), this->getConfig(), this->getDepot(), id);
        this->getPairing().push_back(pairing);
    }

}

void Subpopulation::pairingRandomly() {

    for_each(this->getPairing().begin(), this->getPairing().end(), [&] (Pairing & pairing) {
        pairing.pairingRandomly();
    });

}

void Subpopulation::pairingAllVsBest() {

    for_each(this->getPairing().begin(), this->getPairing().end(), [&] (Pairing & pairing) {
        pairing.pairingAllVsBest();
    });

}

void Subpopulation::printPairing() {

    cout << "\n\nSubpopulation::printPairing(): Dep: " << this->getDepot();
    cout << "\tBest: " << this->getBest().getTotalCost() << endl;

    for_each(this->getPairing().begin(), this->getPairing().end(), [&] (Pairing & pairing) {
        pairing.print();
    });

}


//void Subpopulation::evaluateIndividual(bool split) {
//
//    for_each(this->getIndividuals().begin(), this->getIndividuals().end(), [&] (Individual& individual) {
//        individual.evaluate(split);
//    });
//    
//}

void Subpopulation::evolve() {
    //this->evolveGPU();
    this->evolveCPU();
}

void Subpopulation::evolveCPU() {

    // Version used in MONO and MULTI environments

    IndividualsGroup offsprings = IndividualsGroup(this->getProblem(), this->getConfig(), this->getDepot());

    if (this->getProblem()->getMonitor().isTerminated())
        return;

    //std::vector<std::thread> threads;

    for (auto ite = this->getIndividualsGroup().getIndividuals().begin(); ite !=
            this->getIndividualsGroup().getIndividuals().end(); ++ite) {

        Individual ind = *ite;
        Lock *lock = this->getProblem()->getMonitor().getLock(this->getDepot(), ind.getId());

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        if (this->getConfig()->getProcessType() == Enum_Process_Type::MULTI_THREAD) {
            lock->wait(false);
        }

        // Lambda
        for (int l = 0; l < this->getConfig()->getNumOffspringsPerParent(); ++l) {

            Individual offspring = ind.evolve();

            if (offspring.getGene().size() != 0)
                offsprings.add(offspring);

            if (this->getProblem()->getMonitor().isTerminated())
                break;

        }

    };

    if (this->getProblem()->getMonitor().isTerminated())
        return;

    this->getIndividualsGroup().shrink(offsprings);

    for (auto ite = this->getIndividualsGroup().getIndividuals().begin(); ite !=
            this->getIndividualsGroup().getIndividuals().end(); ++ite) {

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        Individual ind = *ite;
        Lock *lock = this->getProblem()->getMonitor().getLock(this->getDepot(), ind.getId());
        ind.setLocked(false);
        lock->notify(true);
    }

}

void Subpopulation::evolveGPU() {

    // Version used in GPU environment

    if (this->getProblem()->getMonitor().isTerminated())
        return;

    //Lock* gpuLock = this->getProblem()->getMonitor().getGPULock();
    //gpuLock->lock();

    IndividualsGroup offsprings = IndividualsGroup(this->getProblem(), this->getConfig(), this->getDepot());
    this->copyTOManaged();

    // Lambda
    for (int l = 0; l < this->getConfig()->getNumOffspringsPerParent(); ++l) {
        CudaSubpop::evolve(this->getProblem(), mngPop, mngPopRes, this->getDepot(),
            this->getConfig()->getNumSubIndDepots(), l);
        copyFROMManaged(offsprings);

        if (this->getProblem()->getMonitor().isTerminated())
            break;
    }

    if (this->getProblem()->getMonitor().isTerminated())
        return;

    deallocateManaged();
    //gpuLock->unlock();

    localSearchOffsprings(offsprings);

    if (this->getProblem()->getMonitor().isTerminated())
        return;

    this->getIndividualsGroup().shrink(offsprings);

    for (auto ite = this->getIndividualsGroup().getIndividuals().begin(); ite !=
        this->getIndividualsGroup().getIndividuals().end(); ++ite) {

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        Individual ind = *ite;
        Lock *lock = this->getProblem()->getMonitor().getLock(this->getDepot(), ind.getId());
        ind.setLocked(false);
        lock->notify(true);
    }

}

/*
 * Private Methods
 */

void Subpopulation::copyTOManaged() {

    cudaStream_t stream = this->getProblem()->getStream(this->getDepot());

    cudaDeviceSynchronize();
    gpuErrchk(cudaMallocManaged(&mngPop, this->getIndividualsGroup().getIndividuals().size() * sizeof(StrIndividual), cudaMemAttachHost));
    gpuErrchk(cudaMallocManaged(&mngPopRes, this->getIndividualsGroup().getIndividuals().size() * sizeof(StrIndividual), cudaMemAttachHost));
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    for (auto i = 0; i < this->getIndividualsGroup().getIndividuals().size(); ++i) {

        mngPop[i].allocate(this->getIndividualsGroup().getIndividuals().at(i).getGene().size(), stream);
        mngPopRes[i].allocate(this->getIndividualsGroup().getIndividuals().at(i).getGene().size(), stream);

        for (int j = 0; j < mngPop[i].length; ++j) {
            mngPop[i].gene[j] = this->getIndividualsGroup().getIndividuals().at(i).getGene().at(j);
            mngPopRes[i].gene[j] = mngPop[i].gene[j];
        }

        mngPop[i].operations = this->getIndividualsGroup().getIndividuals().at(i).autoUpdate(false);
        mngPopRes[i].operations = mngPop[i].operations;

    }

}

void Subpopulation::copyFROMManaged(IndividualsGroup& offsprings) {

    for (int i = 0; i < this->getConfig()->getNumSubIndDepots(); ++i) {
        Individual ind = this->getIndividualsGroup().getIndividuals().at(i).copy(false);
        for (int j = 0; j < mngPopRes[i].length; ++j)
            ind.add(mngPopRes[i].gene[j]);

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        ind.evaluate(true);
        offsprings.add(ind);
    }

}

void Subpopulation::localSearchOffsprings(IndividualsGroup& offsprings) {

    float cost;

    for (auto i = 0; i < offsprings.getIndividuals().size(); ++i) {

        cost = offsprings.getIndividuals().at(i).getTotalCost();

        if (Random::randFloat() <= offsprings.getIndividuals().at(i).getMutationRatePLS()) {

            offsprings.getIndividuals().at(i).localSearch();
            offsprings.getIndividuals().at(i).routesToGenes();
        }

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        if (offsprings.getIndividuals().at(i).getRoutes().size() == 0) {
            offsprings.getIndividuals().at(i).evaluate(true);
        }

        if (Util::isBetterSolution(offsprings.getIndividuals().at(i).getTotalCost(), cost)) {
            offsprings.getIndividuals().at(i).updateParameters(true);
        }
        else {
            offsprings.getIndividuals().at(i).updateParameters(false);
        }

        if (this->getProblem()->getMonitor().isTerminated())
            break;

    }
}

void Subpopulation::deallocateManaged() {

    for (int i = 0; i < this->getConfig()->getNumSubIndDepots(); ++i) {
        mngPop[i].deallocate();
        mngPopRes[i].deallocate();
    }

    cudaFree(mngPop);
    cudaFree(mngPopRes);

}