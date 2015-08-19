/*
 * File:   Community.cpp
 * Author: fernando
 *
 * Created on July 21, 2014, 9:57 PM
 */

#include "Community.hpp"
#include "PathRelinking.hpp"

/*
 * Constructors
 */

Community::Community(MDVRPProblem* problem, AlgorithmConfig* config, EliteGroup *eliteGroup) :
problem(problem), config(config), eliteGroup(eliteGroup) {

    this->createSubpopulations();

}

Community::~Community() {
}

/*
 * Getters and Setters
 */

vector<Subpopulation>& Community::getSubpops() {
    return this->subpops;
}

void Community::setSubpops(vector<Subpopulation> subpops) {
    //this->subpops = subpops;
}

vector<Subpopulation>
Community::getSubpopsConst() const {
    return this->subpops;
}

vector<Subpopulation>&
Community::getSubpopsPool() {
    return subpopsPool;
}

void
Community::setSubpopsPool(vector<Subpopulation> subpopsPool) {
    //this->subpopsPool = subpopsPool;
}

EliteGroup*
Community::getEliteGroup() {
    return this->eliteGroup;
}

void
Community::setEliteGroup(EliteGroup* eliteGroup) {
    this->eliteGroup = eliteGroup;
}

AlgorithmConfig*
Community::getConfig() const {
    return this->config;
}

void
Community::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem*
Community::getProblem() const {
    return this->problem;
}

void
Community::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

forward_list<typedef_evolution>&
Community::getEvolution() {
    return evolution;
}

void Community::setEvolution(forward_list<typedef_evolution> evolution) {
    this->evolution = evolution;
}

/*
 * Methods
 */

void Community::createSubpopulations() {

    for (int depot = 0; depot < this->getProblem()->getDepots(); ++depot) {
        Subpopulation subpop = Subpopulation(this->getProblem(), this->getConfig(), this->getEliteGroup(), depot);
        subpop.createIndividuals();
        this->getSubpops().push_back(subpop);
    }

    //this->setEliteGroup(EliteGroup(this->getProblem(), this->getConfig()));

}

void Community::pairingRandomly() {

    for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpops) {
        subpops.pairingRandomly();
    });

}

void Community::pairingAllVsBest() {

    for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpops) {
        subpops.pairingAllVsBest();
    });

}

void Community::printPairing() {

    for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpops) {
        subpops.printPairing();
    });

}

void Community::evaluateSubpops(bool firstEvaluation) {

    if (firstEvaluation || this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD)
        this->evaluateSubpopsAndUpdateEG(firstEvaluation);
    else
        this->evaluateSubpopsParallel();

}

void Community::printEvolution() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        float gap = Util::calculateGAP(this->getEliteGroup()->getBest().getTotalCost(),
            this->getProblem()->getBestKnowSolution());

        printf("%lu; %.2f; %.2f; %.2f; %.2f\n", this->getProblem()->getMonitor().getGeneration(),
            Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()),
            this->getEliteGroup()->getBest().getTotalCost(),
            this->getProblem()->getBestKnowSolution(),
            gap);
    }
    else {

        for_each(this->getEvolution().begin(), this->getEvolution().end(), [&](typedef_evolution & evol) {
            float gap = Util::calculateGAP(evol.cost,
                this->getProblem()->getBestKnowSolution());

            printf("%.2f; %.2f; %.2f; %.2f\n", evol.time, evol.cost,
                this->getProblem()->getBestKnowSolution(),
                gap);
        });


    }

}

bool Community::isStopCriteriaMet() {

    bool result = false;

    // Numero de geracoes
    if (this->getConfig()->getStopCriteria() == NUM_GER) {
        if (this->getProblem()->getMonitor().getGeneration() > this->getConfig()->getNumGen()) {
            result = true;
        }
    }
    else {

        // Tempo de processamento OU tempo sem atualizacao
        if (Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()) >= this->getConfig()->getExecutionTime() ||
            (this->getConfig()->getMaxTimeWithoutUpdate() > 0 &&
            Util::diffTimeFromStart(this->getProblem()->getMonitor().getLastTimeUpdate()) >= this->getConfig()->getMaxTimeWithoutUpdate())) {
            result = true;
            //cout << "Encerrado!";
        }
    }

    this->getProblem()->getMonitor().setTerminated(result);
    return result;

}

void Community::writeLogToFile() {

    float cost = this->getEliteGroup()->getBest().getTotalCost();
    float gap = Util::calculateGAP(cost, this->getProblem()->getBestKnowSolution());

    FILE *fp;
    fp = fopen(LOG_RUN_FILE, "a");

    // Instancia;
    fprintf(fp, "%s;", this->getProblem()->getInstCode().c_str()); // c_str());

    if (this->getConfig()->isWriteFactors()) {

        // pop_size; exec_time; MaxTimeWithoutUpdate; mutation_rate; elite_size;
        fprintf(fp, "%d;", this->getConfig()->getNumSubIndDepots());
        fprintf(fp, "%.2f;", this->getConfig()->getExecutionTime());
        fprintf(fp, "%.2f;", this->getConfig()->getMaxTimeWithoutUpdate());
        fprintf(fp, "%.2f;", this->getConfig()->getMutationRatePLS());
        fprintf(fp, "%d;", this->getConfig()->getEliteGroupLimit());

    }

    // geracoes; tempo; update_time; coevolMDVRP; minimo; gap       
    //    fprintf(fp, "%lu;%.2f;%.2f;%.2f;%.2f;%.2f\n", this->getProblem()->getMonitor().getGeneration(),
    //            Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()),
    //            difftime( this->getProblem()->getMonitor().getLastTimeUpdate(), this->getProblem()->getMonitor().getStart() ),
    //            cost, this->getProblem()->getBestKnowSolution(), gap);

    // process_time; last_update; ESCoevolMDVRP; bks; GAP
    fprintf(fp, "%.2f;%.2f;%.2f;%.2f;%.2f\n", Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()),
        difftime(this->getProblem()->getMonitor().getLastTimeUpdate(), this->getProblem()->getMonitor().getStart()),
        cost, this->getProblem()->getBestKnowSolution(), gap);

    fclose(fp);

}

void Community::evolve() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        for (auto ite = this->getSubpops().begin(); ite != this->getSubpops().end(); ++ite) {
            (*ite).evolve();

            if (this->getProblem()->getMonitor().isTerminated())
                break;
        }

    }
    else {

        // Copy the first values
        //this->setSubpopsPool(this->getSubpops());

        while (this->getProblem()->getMonitor().isTerminated() == false) {

            if (!this->getProblem()->getMonitor().isStarted())
                continue;

            if (this->getProblem()->getMonitor().isCopyFromSubpopsRequested()) {

                while (this->getProblem()->getMonitor().isEvaluatingSolutions()
                    || this->getProblem()->getMonitor().isUpdatingEliteGroup()
                    || this->getProblem()->getMonitor().isEliteGroupLocalSearch()) {
                    //printf("Waiting...\n");

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;

                    continue;
                }

                //this->getProblem()->getMonitor().addToLog("Start: isCopyFromSubpopsRequested...");

                // Get paring values from pool
                /*
                for (auto id = 0; id < this->getSubpopsPool().size(); ++id) {
                //printf("Updating subpop id = %d\n", id);
                this->getSubpops().at(id).setPairing(this->getSubpopsPool().at(id).getPairing());
                }
                */

                //--this->getProblem()->getMonitor().getMutexLocker().lock();

                this->getSubpopsPool().clear();

                //printf("Copiando pool\n");
                //this->setSubpopsPool(this->getSubpopsConst());
                //printf("Copiado pool\n");
                this->getProblem()->getMonitor().setCopyFromSubpopsAllowed(true);
                this->getProblem()->getMonitor().setCopyFromSubpopsRequested(false);

                //--this->getProblem()->getMonitor().getMutexLocker().unlock();

                //this->getProblem()->getMonitor().addToLog("End: isCopyFromSubpopsRequested.");
            }

            if (this->getProblem()->getMonitor().isTerminated())
                break;

            this->getProblem()->getMonitor().setEvolvingSubpops(true);
            //this->getProblem()->getMonitor().addToLog("Start: Evolving Subpops...");

            std::vector<std::thread> threads;

            for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpop) {

                threads.push_back(std::thread([&subpop]() {
                    try {
                        Lock *subpoplock = subpop.getProblem()->getMonitor().getSubpopLock(subpop.getDepot());
                        //printf("evolve: Waiting subpop: %d\n", subpop.getDepot());
                        subpoplock->wait(false);
                        //printf("evolve: Evolving subpop: %d\n", subpop.getDepot());
                        subpop.evolve();
                        //printf("evolve: Notifying subpop: %d\n", subpop.getDepot());
                        subpoplock->notify(true);
                    }
                    catch (exception& e) {
                        cout << e.what() << endl;
                    }
                }));

            });

            for (auto& th : threads)
                th.join();

            this->getProblem()->getMonitor().setEvolvingSubpops(false);
            //this->getProblem()->getMonitor().addToLog("End: Evolving Subpops.");

        }
    }

}

void Community::evaluate() {

    std::vector<std::thread> threads;

    for (int s = 0; s < this->getProblem()->getDepots(); ++s) {

        threads.push_back(std::thread([s, this]() {
            //printf("Starting %d\n", s);
            this->evaluateSubpop(s);
        }));

    }

    //printf("Threads: %lu\n", threads.size());

    for (auto& th : threads)
        th.join();

}

void Community::updateBestIndividuals() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        for (auto depot = 0; depot < this->getSubpops().size(); ++depot) {
            this->getSubpops().at(depot).setBest(this->getEliteGroup()->getBest().getIndividuals().at(depot));
        }

    }
    else {

        while (this->getProblem()->getMonitor().isTerminated() == false) {

            if (!this->getProblem()->getMonitor().isStarted())
                continue;

            if (this->getProblem()->getMonitor().isUpdatingEliteGroup()
                || this->getProblem()->getMonitor().isEliteGroupLocalSearch()
                || !this->getProblem()->getMonitor().isCopyBestIndsRequested())
                continue;

            this->getProblem()->getMonitor().setCopyBestIndsRequested(false);

            this->getProblem()->getMonitor().setUpdatingBestInds(true);
            //this->getProblem()->getMonitor().addToLog("Start: Update Best Individuals...");

            //--this->getProblem()->getMonitor().getMutexLocker().lock();

            for (auto depot = 0; depot < this->getSubpopsPool().size(); ++depot) {
                this->getSubpopsPool().at(depot).setBest(this->getEliteGroup()->getBest().getIndividuals().at(depot));
            }

            //--this->getProblem()->getMonitor().getMutexLocker().unlock();

            this->getProblem()->getMonitor().setCopyBestIndsAllowed(true);
            this->getProblem()->getMonitor().setUpdatingBestInds(false);

            //this->getProblem()->getMonitor().addToLog("End: Update Best Individuals.");
        }
    }

}

void Community::manager() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {

        this->getProblem()->getMonitor().setStarted(true);

        // While stop criteria is not met
        while (this->isStopCriteriaMet() == false) {

            this->evolve();
            this->evaluateSubpops(false);
            this->getEliteGroup()->localSearch();
            this->updateBestIndividuals();
            this->checkEvolution();

        }

    }
    else {

        std::vector<std::thread> threads;

        this->getProblem()->getMonitor().setStarted(false);

        threads.push_back(std::thread([&]() {

            this->getProblem()->getMonitor().addToLog("Start thread: Subpops Evolve...");
            this->evolve();
            this->getProblem()->getMonitor().addToLog("End thread: Subpops Evolve.");

        }));

        threads.push_back(std::thread([&]() {

            this->getProblem()->getMonitor().addToLog("Start thread: Complete Solutions Evaluate...");
            this->evaluate();
            this->getProblem()->getMonitor().addToLog("End thread: Complete Solutions Evaluate.");

        }));

        threads.push_back(std::thread([&]() {

            this->getProblem()->getMonitor().addToLog("Start thread: Elite Group Manager...");
            this->getEliteGroup()->manager();
            this->getProblem()->getMonitor().addToLog("End thread: Elite Group Manager.");

        }));

        //        threads.push_back(std::thread([&]() {
        //
        //            this->getProblem()->getMonitor().addToLog("Start thread: Update Best Individuals...");
        //            this->updateBestIndividuals();
        //                    this->getProblem()->getMonitor().addToLog("End thread: Update Best Individuals.");
        //
        //        }));

        threads.push_back(std::thread([&]() {

            this->getProblem()->getMonitor().addToLog("Start thread: Check Evolution...");
            this->checkEvolution();
            this->getProblem()->getMonitor().addToLog("End thread: Check Evolution.");

        }));

        this->getProblem()->getMonitor().setStarted(true);

        // While stop criteria is not met
        this->getProblem()->getMonitor().addToLog("Start thread: Monitor...");
        while (this->isStopCriteriaMet() == false) {
        }
        this->getProblem()->getMonitor().addToLog("End thread: Monitor.");

        for (auto& th : threads)
            th.join();

        this->getProblem()->getMonitor().printLog();

        this->printEvolution();

        //cout << endl << "Elite group pool: " << this->getEliteGroup()->getPool().size() << endl;

    }

}

void Community::printSubpopList() {

    for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpop) {
        printf("\nSubpop ID = %d\n", subpop.getDepot());
        subpop.getIndividualsGroup().printList();
    });

}

/* Private methods */

void Community::checkEvolution() {

    if (this->getConfig()->getProcessType() == Enum_Process_Type::MONO_THREAD) {
        // Print evolution
        if (this->getConfig()->isDisplay() || (this->getProblem()->getMonitor().getGeneration() % 100) == 0)
            // Print evolution
            this->printEvolution();

        this->getProblem()->getMonitor().updateGeneration();

    }
    else {

        float cost = this->getEliteGroup()->getBest().getTotalCost();
        time_t readTime;
        time(&readTime);

        typedef_evolution evolution;
        evolution.cost = cost;
        evolution.time = Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart());

        auto lastPosition = this->getEvolution().insert_after(this->getEvolution().before_begin(), evolution);

        while (this->getProblem()->getMonitor().isTerminated() == false) {

            if (!this->getProblem()->getMonitor().isStarted())
                continue;

            if (Util::diffTimeFromStart(readTime) >= 1) {

                float newCost = this->getEliteGroup()->getBest().getTotalCost();

                if (Util::isBetterSolution(newCost, cost)) {
                    cost = newCost;

                    typedef_evolution evolution;
                    evolution.cost = cost;
                    evolution.time = Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart());

                    lastPosition = this->getEvolution().insert_after(lastPosition, evolution);

                    //cout << evolution.time << "\t" << evolution.cost << endl;

                    //cout << Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()) << ": " 
                    //    << cost << " => " << newCost << endl;

                    printf("%.2f; %.2f; %.2f; %.2f\n", evolution.time, evolution.cost,
                        this->getProblem()->getBestKnowSolution(),
                        Util::calculateGAP(cost, this->getProblem()->getBestKnowSolution()));

                }

                //printf("%.2f; %.2f; %.2f; %.2f\n", Util::diffTimeFromStart(this->getProblem()->getMonitor().getStart()), cost,
                //        this->getProblem()->getBestKnowSolution(),
                //        Util::calculateGAP(cost, this->getProblem()->getBestKnowSolution()));

                time(&readTime);

            }

        }

    }
}

void Community::evaluateSubpopsAndUpdateEG(bool firstEvaluation) {

    for_each(this->getSubpops().begin(), this->getSubpops().end(), [&](Subpopulation & subpop) {

        //subpop.setLocked(true);

        for_each(subpop.getPairing().begin(), subpop.getPairing().end(), [&](Pairing & pairing) {

            IndividualsGroup individuals = IndividualsGroup(this->getProblem(), this->getConfig(), subpop.getDepot());

            // Copy individuals           
            for (size_t depot = 0; depot < pairing.getDepotRelation().size(); ++depot) {

                Individual ind = Individual(this->getProblem(), this->getConfig(), depot, -1);

                if (pairing.getDepotRelation().at(depot) < 0) {
                    // Best
                    ind = this->getSubpops().at(depot).getBest();
                }
                else {
                    ind = this->getSubpops().at(depot).getIndividualsGroup().getIndividuals().at(pairing.getDepotRelation().at(depot));
                }

                individuals.add(ind);

            }

            try {
                // Remove conflicts and Evaluate
                //individuals.printSolution();
                individuals.evaluate(true, true);
            }
            catch (exception& e) {
                cout << e.what() << endl;
            }


            // Without local search at first - to be fast!
            if (!firstEvaluation && Random::randFloat() <= this->getConfig()->getMutationRatePLS()) {
                individuals.localSearch();
            }

            // Select - compare with parent
            if (Util::isBetterSolution(individuals.getTotalCost(), pairing.getCost())) {
                // Replace current

                int id = pairing.getDepotRelation().at(subpop.getDepot());

                pairing.setCost(individuals.getTotalCost());
                subpop.getIndividualsGroup().getIndividuals().at(id) = individuals.getIndividuals().at(subpop.getDepot());
            }

            //scout << subpops.getBest().getTotalCost() << endl;

            // Compare with best of subpopulation
            if (Util::isBetterSolution(individuals.getTotalCost(), subpop.getBest().getTotalCost())) {
                subpop.setBest(individuals.getIndividuals().at(subpop.getDepot()));
            }

            /*
             * Elite group
             */
            this->getEliteGroup()->update(individuals);

        });

        //subpop.setLocked(false);

    });

}

void
Community::evaluateSubpopsParallelOLD() {

    while (this->getProblem()->getMonitor().isTerminated() == false) {

        //this->getProblem()->getMonitor().addToLog("Checking: evaluateSubpopsParallel().");

        // Wait until ES is updated
        //if (this->getProblem()->getMonitor().isUpdatingEliteGroup() 
        //        || this->getProblem()->getMonitor().isEliteGroupLocalSearch())
        //    continue;

        // Request copy of subpops
        this->getProblem()->getMonitor().setCopyFromSubpopsRequested(true);

        // Wait until copy is allowed
        while (!this->getProblem()->getMonitor().isCopyFromSubpopsAllowed()
            //|| this->getProblem()->getMonitor().isEliteGroupLocalSearch()
            || this->getProblem()->getMonitor().isUpdatingEliteGroup()) {
            //|| this->getEliteGroup()->getPool().size() > 0) {

            if (this->getProblem()->getMonitor().isTerminated())
                break;

            continue;
        }

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        // Request update from elite group - best individuals in each subpop
        this->getProblem()->getMonitor().setCopyBestIndsRequested(true);

        // Wait until copy is allowed
        while (!this->getProblem()->getMonitor().isCopyBestIndsAllowed()
            //|| this->getProblem()->getMonitor().isEliteGroupLocalSearch()
            || this->getProblem()->getMonitor().isUpdatingEliteGroup()
            || this->getProblem()->getMonitor().isUpdatingBestInds())
            continue;

        this->getProblem()->getMonitor().setCopyBestIndsRequested(false);
        this->getProblem()->getMonitor().setCopyFromSubpopsAllowed(false);
        this->getProblem()->getMonitor().setEvaluatingSolutions(true);

        //this->getProblem()->getMonitor().addToLog("Start: evaluateSubpopsParallel()...");

        std::vector<std::thread> threads;

        //for_each(this->getSubpopsPool().begin(), this->getSubpopsPool().end(), [&] (Subpopulation & subpop) {
        //    for_each(subpop.getPairing().begin(), subpop.getPairing().end(), [&] (Pairing & pairing) {

        for (auto sId = 0; sId < this->getSubpopsPool().size(); ++sId) {

            for (auto pId = 0; pId < this->getSubpopsPool().at(sId).getPairing().size(); ++pId) {

                //printf("Start Eval: S = %d\tP = %d\n", sId, pId);

                IndividualsGroup individuals = IndividualsGroup(this->getProblem(), this->getConfig(), this->getSubpopsPool().at(sId).getDepot());

                // Copy individuals           
                for (size_t depot = 0; depot < this->getSubpopsPool().at(sId).getPairing().at(pId).getDepotRelation().size(); ++depot) {

                    Individual ind = Individual(this->getProblem(), this->getConfig(), depot, -1);

                    if (this->getSubpopsPool().at(sId).getPairing().at(pId).getDepotRelation().at(depot) < 0) {
                        // Best
                        ind = this->getSubpopsPool().at(depot).getBest();
                    }
                    else {
                        ind = this->getSubpopsPool().at(depot).getIndividualsGroup().getIndividuals().at(this->getSubpopsPool().at(sId).getPairing().at(pId).getDepotRelation().at(depot));
                    }

                    individuals.add(ind);

                }

                /*
                 * Insert in Elite group pool
                 */
                this->getEliteGroup()->getPool().push_back(individuals);
            }
        }

        for (auto idInd = 0; idInd < this->getEliteGroup()->getPool().size(); ++idInd) {

            //IndividualsGroup *individuals = &this->getEliteGroup()->getPool().at(idInd);

            // Run in thread
            threads.push_back(std::thread([this, idInd]() {

                try {
                    // Remove conflicts and Evaluate
                    //individuals.printSolution();
                    this->getEliteGroup()->getPool().at(idInd).evaluate(true, false);
                }
                catch (exception& e) {
                    cout << e.what() << endl;
                    this->getEliteGroup()->getPool().at(idInd).printSolution();
                    cout << this->getEliteGroup()->getPool().size();
                }

                if (Random::randFloat() <= this->getConfig()->getMutationRatePLS()) {
                    //this->getEliteGroup()->getPool().at(idInd).localSearch(1);
                }

                // Select - compare with its parent
                //            if (Util::isBetterSolution(individuals.getTotalCost(), this->getSubpopsPool().at(sId).getPairing().at(pId).getCost())) {
                //                // Replace current
                //                //int id = this->getSubpopsPool().at(sId).getPairing().at(pId).getDepotRelation().at(this->getSubpopsPool().at(sId).getDepot());
                //                this->getSubpopsPool().at(sId).getPairing().at(pId).setCost(individuals.getTotalCost());
                //                //this->getSubpopsPool().at(sId).getIndividualsGroup().getIndividuals().at(id) = individuals.getIndividuals().at(this->getSubpopsPool().at(sId).getDepot());
                //            }

                //printf("End Eval: S = %d\tP = %d\n", sId, pId);

            }));

            // Compare with best of subpopulation
            //if (Util::isBetterSolution(individuals.getTotalCost(), subpop.getBest().getTotalCost())) {
            //    subpop.setBest(individuals.getIndividuals().at(subpop.getDepot()));
            //}


        }

        //subpop.setLocked(false);

        //}

        for (auto& th : threads)
            th.join();

        this->getProblem()->getMonitor().setEvaluatingSolutions(false);
        this->getProblem()->getMonitor().setUpdateEGRequested(true);

        //this->getProblem()->getMonitor().addToLog("End: evaluateSubpopsParallel().");

    }

}

void Community::evaluateSubpopsParallel() {

    while (this->getProblem()->getMonitor().isTerminated() == false) {

        std::vector<std::thread> threads;

        for (typedef_vectorIntSize idSubpop = 0; idSubpop < this->getSubpops().size(); ++idSubpop) {

            printf("evaluateSubpopsParallel: Waiting subpop: %lu\n", idSubpop);
            this->getProblem()->getMonitor().getSubpopLock(idSubpop)->wait(true);
            printf("evaluateSubpopsParallel: Evaluating subpop: %lu\n", idSubpop);

            for (typedef_vectorIntSize idInd = 0; idInd < this->getSubpops().at(idSubpop).getIndividualsGroup().getIndividuals().size(); ++idInd) {

                if (this->getSubpops().at(idSubpop).getIndividualsGroup().getIndividuals().at(idInd).getRoutes().empty()) {
                    //cout << "Community::evaluateSubpopsParallel() => routes vazio!\n";
                    continue;
                }

                // Run in thread
                threads.push_back(std::thread([this, idSubpop, idInd]() {

                    Individual *ind = &this->getSubpops().at(idSubpop).getIndividualsGroup().getIndividuals().at(idInd);

                    //while (this->getProblem()->getMonitor().isTerminated() == false) {

                    //this->getProblem()->getMonitor().getLock(idSubpop, idInd)->wait(true);

                    if (!ind->isLocked()) {

                        ind->setLocked(true);

                        //printf("Starting Ind: %d-%d\n", ind->getDepot(), ind->getId());

                        //printf("Evolving Ind = D: %d - Id: %d => PM: %.2f / PLS: %.2f / RS: %d\n", ind->getDepot(),
                        //ind->getId(), ind->getMutationRatePM(), ind->getMutationRatePLS(), ind->isRestartMoves());


                        //for_each(subpop.getPairing().begin(), subpop.getPairing().end(), [&] (Pairing & pairing) {

                        IndividualsGroup individuals = IndividualsGroup(this->getProblem(), this->getConfig(), ind->getDepot());

                        // Copy individuals           
                        for (int depot = 0; depot < this->getProblem()->getDepots(); ++depot) {

                            if (ind->getDepot() != depot) {
                                // Best

                                while (1) {

                                    if (this->getProblem()->getMonitor().getMutexLocker().try_lock()) {
                                        Individual partner = this->getEliteGroup()->getBest().getIndividuals().at(depot);
                                        individuals.add(partner);
                                        this->getProblem()->getMonitor().getMutexLocker().unlock();
                                        break;
                                    }

                                    if (this->getProblem()->getMonitor().isTerminated())
                                        break;
                                }

                            }
                            else {
                                individuals.add(*ind);
                            }

                            if (this->getProblem()->getMonitor().isTerminated())
                                break;


                        }

                        individuals.evaluate(true, false);

                        if (Random::randFloat() <= this->getConfig()->getMutationRatePLS()) {

                            individuals.localSearch();

                            //PathRelinking path = PathRelinking(this->getProblem(), this->getConfig());
                            //IndividualsGroup best = this->getEliteGroup()->getBest();
                            //path.operate(individuals, best);

                        }

                        bool lock = this->getProblem()->getMonitor().forceLock();

                        if (lock) {

                            //if (this->getProblem()->getMonitor().getMutexLocker().try_lock()) {
                            //printf("Lock(): Community::evaluateSubpopsParallel()\n");
                            this->getEliteGroup()->update(individuals);
                            this->getProblem()->getMonitor().getMutexLocker().unlock();
                            //printf("Unlock(): Community::evaluateSubpopsParallel()\n");

                            //                            } else {
                            //                                this->getTrash().push_back(individuals);
                        }

                        ind->setLocked(false);
                        //this->getProblem()->getMonitor().getLock(idSubpop, idInd)->notify(false);


                    }

                    //}
                }));

                //});

                //});
            }

            printf("evaluateSubpopsParallel: Notifying subpop: %lu\n", idSubpop);
            this->getProblem()->getMonitor().getSubpopLock(idSubpop)->notify(false);

        }

        for (auto& th : threads)
            th.join();

    }

}

void Community::evaluateSubpop(int subpopID) {

    while (this->getProblem()->getMonitor().isTerminated() == false) {

        if (!this->getProblem()->getMonitor().isStarted())
            continue;

        std::vector<std::thread> threads;

        //printf("evaluateSubpop: Waiting subpop: %d\n", subpopID);
        this->getProblem()->getMonitor().getSubpopLock(subpopID)->wait(true);
        //printf("evaluateSubpop: Evaluating subpop: %d\n", subpopID);

        if (this->getProblem()->getMonitor().isTerminated())
            break;

        for (typedef_vectorIntSize idInd = 0; idInd < this->getSubpops().at(subpopID).getIndividualsGroup().getIndividuals().size(); ++idInd) {

            if (this->getProblem()->getMonitor().isTerminated())
                break;

            // Run in thread
            threads.push_back(std::thread([this, subpopID, idInd]() {

                bool indReady = true;

                this->getProblem()->getMonitor().getLock(subpopID, idInd)->wait(true);
                //printf("evaluateSubpop: Evaluating subpop: %d - Ind: %lu\n", subpopID, idInd);

                if (this->getProblem()->getMonitor().isTerminated()) {
                    return;
                }

                //Individual ind2 = this->getSubpops().at(subpopID).getIndividualsGroup().getIndividuals().at(idInd);
                //Individual *ind = &ind2;
                Individual *ind = &this->getSubpops().at(subpopID).getIndividualsGroup().getIndividuals().at(idInd);
                IndividualsGroup individuals = IndividualsGroup(this->getProblem(), this->getConfig(), ind->getDepot());

                // Copy individuals           
                for (int depot = 0; depot < this->getProblem()->getDepots(); ++depot) {

                    if (ind->getDepot() != depot) {
                        // Best

                        //while (1) {

                        bool lock = this->getProblem()->getMonitor().forceLock();

                        if (lock) {
                            Individual partner = this->getEliteGroup()->getBest().getIndividuals().at(depot);
                            individuals.add(partner);
                            this->getProblem()->getMonitor().getMutexLocker().unlock();
                            //break;
                        }
                        else {
                            indReady = false;
                            break;
                        }

                        if (this->getProblem()->getMonitor().isTerminated())
                            break;


                    }
                    else {
                        individuals.add(*ind);
                    }

                    if (this->getProblem()->getMonitor().isTerminated())
                        break;


                }

                if (!indReady)
                    return;

                if (this->getProblem()->getMonitor().isTerminated()) {
                    return;
                }

                individuals.evaluate(true, false);

                if (this->getProblem()->getMonitor().isTerminated()) {
                    return;
                }

                if (Random::randFloat() <= this->getConfig()->getMutationRatePLS()) {
                    individuals.localSearch();
                }

                if (this->getProblem()->getMonitor().isTerminated()) {
                    return;
                }

                bool lock = this->getProblem()->getMonitor().forceLock();

                if (this->getProblem()->getMonitor().isTerminated()) {
                    return;
                }

                if (lock) {
                    this->getEliteGroup()->update(individuals);
                    this->getProblem()->getMonitor().getMutexLocker().unlock();
                }

                ind->setLocked(false);
                this->getProblem()->getMonitor().getLock(subpopID, idInd)->notify(false);
                //printf("evaluateSubpop: END Evaluating subpop: %d - Ind: %lu\n", subpopID, idInd);

            }));
        }

        for (auto& th : threads) {
            //cout << "Join thread " << th.get_id() << endl;
            th.join();
        }

        //printf("evaluateSubpopsParallel: Notifying subpop: %d\n", subpopID);
        this->getProblem()->getMonitor().getSubpopLock(subpopID)->notify(false);

    }

}
