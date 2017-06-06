/*
 * File:   EliteGroup.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 24, 2014, 11:40 AM
 */

#include "EliteGroup.hpp"
#include "PathRelinking.hpp"

/*
 * Constructors and Destructor
 */

EliteGroup::EliteGroup() {
    this->setChanged(true);
    //this->setLocked(false);
}

EliteGroup::EliteGroup(MDVRPProblem *problem, AlgorithmConfig *config) :
problem(problem), config(config) {
    this->setBest(IndividualsGroup(this->getProblem(), this->getConfig(), -1));
    this->setChanged(true);
    this->setLocked(false);
}

/*
 * Getters and Setters
 */

vector<IndividualsGroup>& EliteGroup::getEliteGroup() {
    return this->eliteGroup;
}

void EliteGroup::setEliteGroup(vector<IndividualsGroup> eliteGroup) {
    this->eliteGroup = eliteGroup;
}

IndividualsGroup& EliteGroup::getBest() {
    return this->best;
}

void EliteGroup::setBest(IndividualsGroup best) {
    this->best = best;
    time_t update;
    time(&update);
    this->getProblem()->getMonitor().setLastTimeUpdate(update);
}

AlgorithmConfig* EliteGroup::getConfig() const {
    return this->config;
}

void EliteGroup::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* EliteGroup::getProblem() const {
    return this->problem;
}

void EliteGroup::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

bool EliteGroup::isChanged() const {
    return changed;
}

void EliteGroup::setChanged(bool changed) {
    this->changed = changed;
}

bool EliteGroup::isLocked() const {
    return locked;
}

void EliteGroup::setLocked(bool locked) {

    //    if (locked)
    //        this->getConfig()->getMutexLocker().lock();
    //    else
    //        this->getConfig()->getMutexLocker().unlock();

    this->locked = locked;
}

vector<IndividualsGroup>& EliteGroup::getPool() {
    return pool;
}

void EliteGroup::setPool(vector<IndividualsGroup> pool) {
    this->pool = pool;
}

/*
 * Public Methods
 */

void EliteGroup::update(IndividualsGroup& individuals) {

    if (this->getEliteGroup().size() < this->getConfig()->getEliteGroupLimit()) {
        this->getEliteGroup().push_back(individuals);
        this->setChanged(true);
    }
    else {

        // Replace individual checking diversification
        // Check if it is better than something
        //for_each (this->getEliteGroup().begin(), this->getEliteGroup().end(), [&] (IndividualsGroup & eliteIndividual) {
        for (auto ite = this->getEliteGroup().begin(); ite != this->getEliteGroup().end(); ++ite) {

            if (Util::isEqualSolution(individuals.getTotalCost(), (*ite).getTotalCost())) {
                break;
            }

            if (Util::isBetterSolution(individuals.getTotalCost(), (*ite).getTotalCost())) {
                //printf("EliteGroup::update(): From %.2f To %.2f;\n", (*ite).getTotalCost(), individuals.getTotalCost());
                *ite = individuals;
                this->setChanged(true);
                break;
            }

        }

    }

    if (Util::isBetterSolution(individuals.getTotalCost(), this->getBest().getTotalCost())) {
        //cout << "Best updated: " << individuals.getTotalCost() << endl;
        //printf("EliteGroup::update(): Best updated: From %.2f To %.2f;\n", this->getBest().getTotalCost(), individuals.getTotalCost());
        this->setBest(individuals);
        this->setChanged(true);
        //individuals.printSolution();
    }

}

void EliteGroup::updatePool(IndividualsGroup& individuals) {

    if (this->getPool().size() < this->getConfig()->getEliteGroupLimit()) {
        this->getPool().push_back(individuals);
        this->setChanged(true);
    }
    else {

        // Replace individual checking diversification
        // Check if it is better than someone
        //for_each (this->getEliteGroup().begin(), this->getEliteGroup().end(), [&] (IndividualsGroup & eliteIndividual) {
        for (auto ite = this->getPool().begin(); ite != this->getPool().end(); ++ite) {

            if (Util::isBetterSolution(individuals.getTotalCost(), (*ite).getTotalCost())) {
                *ite = individuals;
                break;
            }

        }

    }

}

void EliteGroup::localSearch() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        // MONO_THREAD version
        for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {

            if (!this->getProblem()->getMonitor().isTerminated()) {
                eliteIndividual.localSearch(false, true);

                if (eliteIndividual.getTotalCost() != this->getBest().getTotalCost()) {

                    PathRelinking path = PathRelinking(this->getProblem(), this->getConfig());
                    path.operate(eliteIndividual, this->getBest());

                }

                eliteIndividual.localSearch(true, true);

                if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {
                    this->setBest(eliteIndividual);
                }
            }

        });

    }
    else {

        this->getProblem()->getMonitor().setEliteGroupLocalSearch(true);

        //printf("Start: EliteGroup::localSearch();\n");
        std::vector<std::thread> threads;

        for (int i = 0; i < (int)this->getPool().size(); ++i) {
            this->getPool().at(i).setId(i);
        }

        runGPUonID = Random::randIntBetween(0, (int)this->getPool().size() - 1);

        // MULTI_THREAD+GPU version
        for_each(this->getPool().begin(), this->getPool().end(), [&](IndividualsGroup & eliteIndividual) {

            if (!this->getProblem()->getMonitor().isTerminated()) {
                threads.push_back(std::thread([&eliteIndividual, this]() {

                    eliteIndividual.localSearch(false, false);

                    if (eliteIndividual.getTotalCost() != this->getBest().getTotalCost()) {
                        PathRelinking path = PathRelinking(this->getProblem(), this->getConfig());
                        path.operate(eliteIndividual, this->getBest());
                    }

                    int id = Random::randIntBetween(0, (int)this->getPool().size() - 1);

                    eliteIndividual.localSearch(true, eliteIndividual.getId() == id);

                }));
            }

        });

        for (auto& th : threads)
            th.join();

        for_each(this->getPool().begin(), this->getPool().end(), [&](IndividualsGroup & eliteIndividual) {
            //this->update(eliteIndividual);            

            if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {

                // Old lock
                bool lock = this->getProblem()->getMonitor().forceLock();

                if (lock) {
                    if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {
                        //--this->getProblem()->getMonitor().getMutexLocker().lock();

                        float gap = Util::calculateGAP(eliteIndividual.getTotalCost(), this->getProblem()->getBestKnowSolution());

                        printf("EliteGroup::localSearch(): Best updated: From %.2f To %.2f (GAP = %.2f);\n", this->getBest().getTotalCost(), eliteIndividual.getTotalCost(), gap);
                        this->setBest(eliteIndividual);
                        this->setChanged(true);
                    }
                    this->getProblem()->getMonitor().getMutexLocker().unlock();
                }
            }

        });

        this->getProblem()->getMonitor().setEliteGroupLocalSearch(false);

        //printf("End: EliteGroup::localSearch();\n");

    }

}

void EliteGroup::manager() {

    while (this->getProblem()->getMonitor().isTerminated() == false) {

        if (!this->getProblem()->getMonitor().isStarted())
            continue;

        if (this->getEliteGroup().size() < this->getConfig()->getEliteGroupLimit())
            continue;

        bool lock = this->getProblem()->getMonitor().forceLock();

        if (lock) {

            //if (this->getProblem()->getMonitor().getMutexLocker().try_lock()) {

            for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {
                this->updatePool(eliteIndividual);
            });

            this->getProblem()->getMonitor().getMutexLocker().unlock();
            this->localSearch();

            int first = (int)this->getEliteGroup().size() / 2;
            //cout << "EliteGroup::manager(): Removing " << (int)this->getEliteGroup().size() - first + 1 << " elite solutions." << endl;
            this->getEliteGroup().erase(this->getEliteGroup().begin() + first, this->getEliteGroup().end());
            this->getPool().clear();

        }

    }
}

void EliteGroup::printValues() {

    cout << "Elite Group values...:\n\n";

    for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {
        printf("=> %.2f\n", eliteIndividual.getTotalCost());
    });

}

void EliteGroup::managerOLD() {

    this->getProblem()->getMonitor().setUpdatingEliteGroup(false);

    while (this->getProblem()->getMonitor().isTerminated() == false) {

        if (!this->getProblem()->getMonitor().isStarted())
            continue;

        if (this->getProblem()->getMonitor().isEvaluatingSolutions()
            || this->getProblem()->getMonitor().isUpdatingBestInds()
            //|| this->getProblem()->getMonitor().isEliteGroupLocalSearch()
            //|| !this->getProblem()->getMonitor().isUpdateEGRequested()
            )
            continue;

        this->getProblem()->getMonitor().setUpdateEGRequested(false);
        this->getProblem()->getMonitor().setUpdatingEliteGroup(true);

        //this->getProblem()->getMonitor().addToLog("Start: EliteGroup::manager()");

        //for_each(this->getPool().begin(), this->getPool().end(), [&] (IndividualsGroup & eliteIndividual) {
        //    this->update(eliteIndividual);
        //});

        //printf("Start EliteGroup::localSearch();\n");
        this->localSearch();
        //printf("End EliteGroup::localSearch();\n");

        //this->getProblem()->getMonitor().addToLog("End: EliteGroup::manager()"); 

        //this->getPool().clear();

        this->getProblem()->getMonitor().setUpdatingEliteGroup(false);

    }
}

void EliteGroup::localSearchOLD() {

    // If there is not change
    if (!isChanged())
        // Go out
        return;

    //if (this->isLocked())
    //    return;

    //this->setLocked(true);

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {

            if (!this->getProblem()->getMonitor().isTerminated()) {
                eliteIndividual.localSearch();

                if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {
                    this->setBest(eliteIndividual);
                }
            }

        });

    }
    else {

        this->getProblem()->getMonitor().setEliteGroupLocalSearch(true);

        std::vector<std::thread> threads;

        for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {

            if (!this->isLocked() && !this->getProblem()->getMonitor().isTerminated()) {
                threads.push_back(std::thread([&eliteIndividual]() {
                    if (!eliteIndividual.isLocked()) {
                        eliteIndividual.setLocked(true);
                        eliteIndividual.localSearch();
                        eliteIndividual.setLocked(false);
                    }
                }));
            }

        });

        for (auto& th : threads)
            th.join();

        //--this->getProblem()->getMonitor().getMutexLocker().lock();

        for_each(this->getEliteGroup().begin(), this->getEliteGroup().end(), [&](IndividualsGroup & eliteIndividual) {

            if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {
                printf("Lock(): EliteGroup::localSearch()\n");

                // Old lock
                bool lock = this->getProblem()->getMonitor().forceLock();

                if (lock) {
                    if (Util::isBetterSolution(eliteIndividual.getTotalCost(), this->getBest().getTotalCost())) {
                        this->setBest(eliteIndividual);
                    }
                    printf("Unlock(): EliteGroup::localSearch()\n");
                    this->getProblem()->getMonitor().getMutexLocker().unlock();
                }
                this->getProblem()->getMonitor().setDoUpdateBest(true);
            }
        });

        //--this->getProblem()->getMonitor().getMutexLocker().unlock();

        this->getProblem()->getMonitor().setEliteGroupLocalSearch(false);

    }

    //this->setLocked(false);

}

/*
 * Private Methods
 */
