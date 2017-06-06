/*
 * File:   LocalSearch.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 28, 2014, 2:28 PM
 */

#include "LocalSearch.hpp"

#define CHECK_DEMAND true

/*
 * Constructors and Destructor
 */

/*
 * Public Methods
 */
bool LocalSearch::processMoveDepotRoute(Route& ru, Route& rv, int move, bool equal, bool gpu, int streamId) {

    Route newRu = ru;
    Route newRv = rv;

    if (ru.getTour().size() == 0)
        return false;

    bool result = operateMoveDepotRouteFacade(newRu, newRv, move, equal, gpu, streamId);

    if (result) {
        newRu.calculateCost();
        if (!equal) newRv.calculateCost();

        if (Util::isBetterSolution(newRu.getTotalCost() + newRv.getTotalCost(),
            ru.getTotalCost() + rv.getTotalCost())) {
            ru = newRu;
            if (!equal) rv = newRv;
            result = true;
        }
        else
            result = false;
    }

    return result;
}

bool LocalSearch::operateMoves(MDVRPProblem* problem, AlgorithmConfig* config, Route& ru, Route& rv, bool equal, bool gpu, int streamId) {

    bool result, improved = false;
    int move;

    vector<int> moves;
    Random::randPermutation(config->getTotalMoves(), 1, moves);

    for (int m = 1; m <= config->getTotalMoves(); ++m) {

        if (config->getLocalSearchType() == RANDOM)
            move = moves.at(m - 1);
        else
            move = m;

        do {
            result = LocalSearch::processMoveDepotRoute(ru, rv, move, equal, gpu, streamId);

            if (result)
                improved = true;

        } while (result == true && !problem->getMonitor().isTerminated());

    }

    return improved;

}

/*
 * Private Methods
 */

bool LocalSearch::operateMoveDepotRouteFacade(Route& ru, Route& rv, int move, bool equal, bool gpu, int streamId) {

    bool result = false;

    //move = 9;

    switch (move) {

        case 1:
            //if (equal || !gpu)
            //    result = operateMoveDepotRouteM1(ru, rv, equal);
            //else {
            //    //Lock* gpuLock = ru.getProblem()->getMonitor().getGPULock();
            //    //gpuLock->lock();
            //    result = cudaLocalSearch(ru, rv, move);
            //    //gpuLock->unlock();
            //}

            result = operateMoveDepotRouteM1(ru, rv, equal);
            break;

        case 2:
            result = operateMoveDepotRouteM2(ru, rv, equal, false);
            break;

        case 3:
            result = operateMoveDepotRouteM3(ru, rv, equal);
            break;

        case 4:
            //if (true || equal || !gpu)
            //    result = operateMoveDepotRouteM4(ru, rv, equal);
            //else {
            //    //Lock* gpuLock = ru.getProblem()->getMonitor().getGPULock();
            //    //gpuLock->lock();
            //    result = cudaLocalSearch(ru, rv, move);
            //    //gpuLock->unlock();
            //}
                
            result = operateMoveDepotRouteM4(ru, rv, equal);
            break;

        case 5:
            result = operateMoveDepotRouteM5(ru, rv, equal);
            break;

        case 6:
            result = operateMoveDepotRouteM6(ru, rv, equal);
            break;

        case 7:
            if (!equal)
                return false;

            if (!gpu)
                result = operateMoveDepotRouteM7(ru, rv, equal);
            else
                result = cudaLocalSearch(ru, rv, move, streamId);

            break;

        case 8:

            if (equal)
                return false;

            result = operateMoveDepotRouteM8(ru, rv, equal);

            if (!gpu)
                result = operateMoveDepotRouteM8(ru, rv, equal);
            else
                result = cudaLocalSearch(ru, rv, move, streamId);

            break;

        case 9:
            if (equal)
                return false;

            result = operateMoveDepotRouteM9(ru, rv, equal);

            if (!gpu)
                result = operateMoveDepotRouteM9(ru, rv, equal);
            else
                result = cudaLocalSearch(ru, rv, move, streamId);

            break;

    }

    return result;
}

// (M1) If u is a customer visit, remove u and place it after v;

bool LocalSearch::operateMoveDepotRouteM1(Route& ru, Route& rv, bool equal) {

    bool result = false;
    bool endProcess = false;
    bool frontInserted = false;
    bool increasePosition = false;
    float cost;

    if (equal) {

        if (ru.getTour().size() <= 1)
            return result;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            Route newRU = ru;
            newRU.remove((*iterRU));

            auto iterRV = newRU.getTour().begin();
            endProcess = false;
            frontInserted = false;

            //newRU.printSolution();

            while (!endProcess) {

                auto position = iterRV;

                increasePosition = true;

                if (!frontInserted || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*iterRV) - 1) == 1) {

                    if (!frontInserted && iterRV == newRU.getTour().begin()) {
                        //cout << "Before: " << *iterRV << endl;
                        position = newRU.addAtFront((*iterRU));
                        frontInserted = true;
                        increasePosition = false;
                    } else {
                        //cout << "After: " << *iterRV << endl;
                        position = newRU.addAfterPrevious(iterRV, (*iterRU));
                        increasePosition = true;
                    }

                    //newRU.printSolution();

                    if (Util::isBetterSolution(newRU.getTotalCost(), ru.getTotalCost())) {
                        ru = newRU;
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        newRU.remove(position);
                        //newRU.printSolution();
                        //cout << endl;
                    }
                }

                if (increasePosition)
                    iterRV++;

                if (iterRV == newRU.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    } else {

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            if (CHECK_DEMAND) {
                if (rv.getProblem()->getDemand().at((*iterRU) - 1) + rv.getDemand() > rv.getCapacity())
                    continue;
            }

            //            // Check service duration time
            //            if (rv.getProblem()->getDuration() > 0)
            //                if (rv.getProblem()->getServiceTime().at((*iterRU) - 1) + rv.getServiceTime() > rv.getDuration())
            //                    continue;

            Route newRU = ru;
            newRU.remove((*iterRU));

            auto iterRV = rv.getTour().begin();
            endProcess = false;
            frontInserted = false;

            while (!endProcess) {

                auto position = iterRV;
                cost = rv.getTotalCost();

                increasePosition = true;

                if (!frontInserted || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*iterRV) - 1) == 1) {

                    if (!frontInserted && iterRV == rv.getTour().begin()) {
                        //cout << "Before: " << *iterRV << endl;
                        position = rv.addAtFront((*iterRU));
                        frontInserted = true;
                        increasePosition = false;
                    } else {
                        //cout << "After: " << *iterRV << endl;
                        position = rv.addAfterPrevious(iterRV, (*iterRU));
                        increasePosition = true;
                    }

                    //rv.printSolution();

                    if (Util::isBetterSolution(newRU.getTotalCost() + rv.getTotalCost(), ru.getTotalCost() + cost)) {
                        ru = newRU;
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        rv.remove(position);
                        //rv.printSolution();
                        //cout << endl;
                    }
                }

                if (increasePosition)
                    iterRV++;

                if (iterRV == rv.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }
    }

    return result;
}

// (M2) If u and x are customer visits, remove them, then place u and x after v;

bool LocalSearch::operateMoveDepotRouteM2(Route& ru, Route& rv, bool equal, bool operateM3) {

    bool result = false;
    bool endProcess = false;
    bool frontInserted = false;
    bool increasePosition = false;
    float cost;
    int demand, customer1, customer2;

    if (equal) {

        if (ru.getTour().size() <= 2)
            return result;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            if (next(iterRU) == ru.getTour().end())
                continue;

            Route newRU = ru;

            auto nextIterRU = next(iterRU);

            //newRU.printSolution();

            newRU.remove((*iterRU));
            newRU.remove((*nextIterRU));
            //newRU.printSolution();

            // place x and u after v;
            if (operateM3) {
                customer1 = (*nextIterRU);
                customer2 = (*iterRU);
            } else { // then place u and x after v;
                customer1 = (*iterRU);
                customer2 = (*nextIterRU);
            }

            auto iterRV = newRU.getTour().begin();
            endProcess = false;
            frontInserted = false;

            while (!endProcess) {

                auto firstPosition = iterRV;
                auto secPosition = iterRV;

                increasePosition = true;

                if (!frontInserted || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*iterRV) - 1) == 1) {

                    if (!frontInserted && iterRV == newRU.getTour().begin()) {
                        //cout << "Before: " << *iterRV << endl;
                        firstPosition = newRU.addAtFront(customer1);
                        frontInserted = true;
                        increasePosition = false;
                    } else {
                        //cout << "After: " << *iterRV << endl;
                        firstPosition = newRU.addAfterPrevious(iterRV, customer1);
                        increasePosition = true;
                    }

                    secPosition = newRU.addAfterPrevious(firstPosition, customer2);
                    //newRU.printSolution();

                    if (Util::isBetterSolution(newRU.getTotalCost(), ru.getTotalCost())) {
                        ru = newRU;
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        newRU.remove(firstPosition);
                        newRU.remove(secPosition);
                        //newRU.printSolution();
                        //cout << endl;
                    }

                }

                if (increasePosition)
                    iterRV++;

                if (iterRV == newRU.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    } else {

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            if (next(iterRU) == ru.getTour().end())
                continue;

            demand = rv.getProblem()->getDemand().at((*iterRU) - 1);
            demand += rv.getProblem()->getDemand().at((*next(iterRU)) - 1);

            if (CHECK_DEMAND) {
                if (demand + rv.getDemand() > rv.getCapacity())
                    continue;
            }
            //            // Check service duration time
            //            if (rv.getProblem()->getDuration() > 0) {
            //
            //                service = rv.getProblem()->getServiceTime().at((*iterRU) - 1);
            //                service += rv.getProblem()->getServiceTime().at((*next(iterRU)) - 1);
            //
            //                if (service + rv.getServiceTime() > rv.getDuration())
            //                    continue;
            //            }

            //ru.printSolution();

            Route newRU = ru;
            auto nextIterRU = next(iterRU);

            newRU.remove((*iterRU));
            newRU.remove((*nextIterRU));
            //newRU.printSolution();

            // place x and u after v;
            if (operateM3) {
                customer1 = (*nextIterRU);
                customer2 = (*iterRU);
            } else { // then place u and x after v;
                customer1 = (*iterRU);
                customer2 = (*nextIterRU);
            }

            //printf("%d\t%d\n", customer1, customer2);

            //cout << "\n\nTest = " << (*iterRU) << endl;

            //for (auto iterRV = newRU.getTour().begin(); iterRV != newRU.getTour().end(); ++iterRV) {
            auto iterRV = rv.getTour().begin();
            endProcess = false;
            frontInserted = false;

            while (!endProcess) {

                auto firstPosition = iterRV;
                auto secPosition = iterRV;

                cost = rv.getTotalCost();

                increasePosition = true;
                if (!frontInserted || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*iterRV) - 1) == 1) {

                    if (!frontInserted && iterRV == rv.getTour().begin()) {
                        //cout << "Before: " << *iterRV << endl;
                        firstPosition = rv.addAtFront(customer1);
                        frontInserted = true;
                        increasePosition = false;
                    } else {
                        //cout << "After: " << *iterRV << endl;
                        firstPosition = rv.addAfterPrevious(iterRV, customer1);
                        increasePosition = true;
                    }

                    secPosition = rv.addAfterPrevious(firstPosition, customer2);

                    //rv.printSolution();

                    if (Util::isBetterSolution(newRU.getTotalCost() + rv.getTotalCost(), ru.getTotalCost() + cost)) {
                        ru = newRU;
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        rv.remove(firstPosition);
                        rv.remove(secPosition);
                        //rv.printSolution();
                        //cout << endl;
                    }

                }

                if (increasePosition)
                    iterRV++;

                if (iterRV == rv.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }
    }

    return result;
}

// (M3) If u and x are customer visits, remove them, then place x and u after v;

bool LocalSearch::operateMoveDepotRouteM3(Route& ru, Route& rv, bool equal) {
    return operateMoveDepotRouteM2(ru, rv, equal, true);
}

// (M4) If u and v are customer visits, swap u and v;

bool LocalSearch::operateMoveDepotRouteM4(Route& ru, Route& rv, bool equal) {

    bool result = false;
    float costBefore, costAfter, b, a;
    int demand;

    if (equal) {

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {
            for (auto iterRV = ru.getTour().begin(); iterRV != ru.getTour().end(); ++iterRV) {

                if ((*iterRU) == (*iterRV))
                    continue;

                //ru.printSolution();
                if (next(iterRV) != ru.getTour().end()) {
                    //  || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*next(iterRV)) - 1) == 1

                    costBefore = ru.calculateCost(iterRU, next(iterRU), demand);
                    costBefore += ru.calculateCost(iterRV, next(iterRV), demand);
                    //b = ru.getTotalCost();
                    std::swap((*iterRU), (*iterRV));

                    costAfter = ru.calculateCost(iterRU, next(iterRU), demand);
                    costAfter += ru.calculateCost(iterRV, next(iterRV), demand);

                    //printf("Change: \t%d\t%d\n", (*iterRU), (*iterRV));
                    //ru.printSolution();

                    //ru.calculateCost();
                    //a = ru.getTotalCost();

                    if (Util::isBetterSolution(costAfter, costBefore)) {
                        //if (Util::isBetterSolution(a, b)) {
                        //ru.calculateCost();
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        std::swap((*iterRU), (*iterRV));
                        //newRU.printSolution();
                        //cout << endl;
                    }
                }

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            //cout << "END\n";

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    } else {

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {
            for (auto iterRV = rv.getTour().begin(); iterRV != rv.getTour().end(); ++iterRV) {

                if (CHECK_DEMAND) {
                    demand = rv.getDemand() - rv.getProblem()->getDemand().at((*iterRV) - 1) + rv.getProblem()->getDemand().at((*iterRU) - 1);
                    if (demand > rv.getCapacity())
                        continue;
                }

                if (CHECK_DEMAND) {
                    demand = ru.getDemand() - ru.getProblem()->getDemand().at((*iterRU) - 1) + ru.getProblem()->getDemand().at((*iterRV) - 1);
                    if (demand > ru.getCapacity())
                        continue;
                }

                //                // Check service duration time
                //                if (rv.getProblem()->getDuration() > 0) {
                //
                //                    service = rv.getServiceTime() - rv.getProblem()->getServiceTime().at((*iterRV) - 1) + rv.getProblem()->getServiceTime().at((*iterRU) - 1);
                //                    if (service > rv.getDuration())
                //                        continue;
                //
                //                    service = ru.getServiceTime() - ru.getProblem()->getServiceTime().at((*iterRU) - 1) + ru.getProblem()->getServiceTime().at((*iterRV) - 1);
                //                    if (service > ru.getDuration())
                //                        continue;
                //
                //                }

                if (next(iterRV) != rv.getTour().end()) {
                    //  || ru.getProblem()->getGranularNeighborhood().at((*iterRU) - 1).at((*next(iterRV)) - 1) == 1

                    //costBefore = ru.calculateCost(iterRU, next(iterRU), demand);
                    //costBefore += rv.calculateCost(iterRV, next(iterRV), demand);
                    b = ru.getTotalCost() + rv.getTotalCost();
                    std::swap((*iterRU), (*iterRV));

                    //costAfter = ru.calculateCost(iterRU, next(iterRU), demand);
                    //costAfter += rv.calculateCost(iterRV, next(iterRV), demand);

                    ru.calculateCost();
                    rv.calculateCost();
                    a = ru.getTotalCost() + rv.getTotalCost();

                    //ru.printSolution();

                    //if (Util::isBetterSolution(costAfter, costBefore)) {
                    if (Util::isBetterSolution(a, b)) {
                        //ru.calculateCost();
                        //rv.calculateCost();
                        result = true;
                        //cout << "Improved \n";
                        break;
                    } else {
                        std::swap((*iterRU), (*iterRV));
                        ru.calculateCost();
                        rv.calculateCost();

                        //newRU.printSolution();
                        //cout << endl;
                    }
                }

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }
    }

    return result;
}

// (M5) If u, x, and v are customer visits, swap u and x with v;

bool LocalSearch::operateMoveDepotRouteM5(Route& ru, Route& rv, bool equal) {

    bool result = false;
    bool endProcess = false;
    bool frontInserted = false;
    bool isBeforeFrontRU = false, isBeforeFrontRV = false;
    float cost;
    int demandU, demandV, customerBeforeRU, customerBeforeRV, customerV, serviceU, serviceV;

    if (equal) {

        if (ru.getTour().size() <= 2)
            return result;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            if (next(iterRU) == ru.getTour().end())
                continue;

            auto beforeIteRU = ru.getTour().begin();
            if (iterRU != ru.getTour().begin())
                beforeIteRU = std::prev(iterRU);

            if (iterRU == ru.getTour().begin()) {
                isBeforeFrontRU = true;
                customerBeforeRU = 0;
            } else {
                isBeforeFrontRU = false;
                customerBeforeRU = (*beforeIteRU);
            }

            Route newRU = ru;

            auto nextIterRU = next(iterRU);

            newRU.remove((*iterRU));
            newRU.remove((*nextIterRU));
            //newRU.printSolution();

            //cout << "Before = " << customerBeforeRU << " - Next = " << (*nextIterRU) << endl;
            //cout << "\n\nTest = " << (*iterRU) << endl;

            //for (auto iterRV = newRU.getTour().begin(); iterRV != newRU.getTour().end(); ++iterRV) {
            auto iterRV = newRU.getTour().begin();
            //cout << "\n\niterRV = " << (*iterRV) << endl;

            endProcess = false;
            frontInserted = false;

            while (!endProcess) {

                auto firstPosition = iterRV;
                auto secPosition = iterRV;

                auto beforeIteRV = iterRV;
                if (beforeIteRV != newRU.getTour().begin())
                    beforeIteRV = prev(iterRV);

                auto insertedRV = iterRV;

                if (iterRV == newRU.getTour().begin()) {
                    isBeforeFrontRV = true;
                    customerBeforeRV = 0;
                } else {
                    isBeforeFrontRV = false;
                    customerBeforeRV = (*beforeIteRV);
                }

                // Remove from V
                customerV = (*iterRV);
                newRU.remove(iterRV);

                // Insert U in V
                firstPosition = newRU.addAfterPrevious(customerBeforeRV, (*iterRU));
                secPosition = newRU.addAfterPrevious(firstPosition, (*nextIterRU));

                //newRU.printSolution();

                // Insert V in U
                if (isBeforeFrontRU)
                    insertedRV = newRU.addAtFront(customerV);
                else if (customerBeforeRU == customerV)
                    insertedRV = newRU.addAfterPrevious((*nextIterRU), customerV);
                else {
                    insertedRV = newRU.addAfterPrevious(customerBeforeRU, customerV);
                }

                //newRU.printSolution();

                if (Util::isBetterSolution(newRU.getTotalCost(), ru.getTotalCost())) {
                    ru = newRU;
                    result = true;
                    //cout << "Improved \n";
                    break;
                } else {
                    newRU.remove(firstPosition);
                    newRU.remove(secPosition);
                    newRU.remove(insertedRV);

                    // Put V in correct place
                    iterRV = newRU.addAfterPrevious(customerBeforeRV, customerV);

                    //newRU.printSolution();
                    //cout << endl;
                }

                //newRU.printSolution();
                //cout << endl << endl;

                iterRV++;

                if (iterRV == newRU.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    } else {

        if (ru.getTour().size() < 2)
            return result;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            if (next(iterRU) == ru.getTour().end())
                continue;

            auto beforeIteRU = ru.getTour().begin();
            if (iterRU != ru.getTour().begin())
                beforeIteRU = std::prev(iterRU);

            if (iterRU == ru.getTour().begin()) {
                isBeforeFrontRU = true;
                customerBeforeRU = 0;
            } else {
                isBeforeFrontRU = false;
                customerBeforeRU = (*beforeIteRU);
            }

            Route newRU = ru;

            auto nextIterRU = next(iterRU);

            newRU.remove((*iterRU));
            newRU.remove((*nextIterRU));
            //newRU.printSolution();

            //cout << "Before = " << customerBeforeRU << " - Next = " << (*nextIterRU) << endl;
            //cout << "\n\nTest = " << (*iterRU) << endl;

            //for (auto iterRV = newRU.getTour().begin(); iterRV != newRU.getTour().end(); ++iterRV) {
            auto iterRV = rv.getTour().begin();

            //cout << "\n\niterRV = " << (*iterRV) << endl;

            endProcess = false;
            frontInserted = false;

            while (!endProcess) {

                demandU = ru.getDemand() - (ru.getProblem()->getDemand().at((*iterRU) - 1) + ru.getProblem()->getDemand().at((*nextIterRU) - 1))
                        + rv.getProblem()->getDemand().at((*iterRV) - 1);

                demandV = rv.getDemand() - rv.getProblem()->getDemand().at((*iterRV) - 1)
                        + (ru.getProblem()->getDemand().at((*iterRU) - 1) + ru.getProblem()->getDemand().at((*nextIterRU) - 1));

                serviceU = 0;
                serviceV = 0;

                //                // Check service duration time
                //                if (rv.getProblem()->getDuration() > 0) {
                //                    // S_ru <- S_ru - (S_u + S_x) + S_v
                //                    serviceU = ru.getServiceTime() -
                //                                (ru.getProblem()->getServiceTime().at((*iterRU) - 1) + ru.getProblem()->getServiceTime().at((*nextIterRU) - 1)) +
                //                                rv.getProblem()->getServiceTime().at((*iterRV) - 1);
                //
                //                    // S_rv <- S_ru - S_v + S_x + S_u
                //                    serviceV = rv.getServiceTime() - rv.getProblem()->getServiceTime().at((*iterRV) - 1) +
                //                            (ru.getProblem()->getServiceTime().at((*iterRU) - 1) + ru.getProblem()->getServiceTime().at((*nextIterRU) - 1));
                //                }

                if (CHECK_DEMAND) {

                    if ((demandU > ru.getCapacity())
                            || (demandV > rv.getCapacity())
                            || serviceU > ru.getDuration()
                            || serviceV > rv.getDuration()) {
                        // Go to next
                        iterRV++;

                        if (iterRV == rv.getTour().end())
                            endProcess = true;

                        if (ru.getProblem()->getMonitor().isTerminated())
                            break;

                        continue;
                    }
                }

                auto firstPosition = iterRV;
                auto secPosition = iterRV;

                auto beforeIteRV = iterRV;
                if (beforeIteRV != rv.getTour().begin())
                    beforeIteRV = prev(iterRV);

                auto insertedRV = iterRV;

                if (iterRV == rv.getTour().begin()) {
                    isBeforeFrontRV = true;
                    customerBeforeRV = 0;
                } else {
                    isBeforeFrontRV = false;
                    customerBeforeRV = (*beforeIteRV);
                }

                cost = rv.getTotalCost();

                // Remove from V
                customerV = (*iterRV);
                rv.remove(iterRV);

                // Insert U in V
                firstPosition = rv.addAfterPrevious(customerBeforeRV, (*iterRU));
                secPosition = rv.addAfterPrevious(firstPosition, (*nextIterRU));

                //rv.printSolution();

                // Insert V in U
                insertedRV = newRU.addAfterPrevious(customerBeforeRU, customerV);

                //newRU.printSolution();

                if (Util::isBetterSolution(newRU.getTotalCost() + rv.getTotalCost(), ru.getTotalCost() + cost)) {
                    ru = newRU;
                    result = true;
                    //cout << "Improved \n";
                    break;
                } else {
                    rv.remove(firstPosition);
                    rv.remove(secPosition);
                    newRU.remove(insertedRV);

                    // Put V in correct place
                    iterRV = rv.addAfterPrevious(customerBeforeRV, customerV);

                }

                iterRV++;

                if (iterRV == rv.getTour().end())
                    endProcess = true;

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }
    }

    return result;
}

// (M6) If u, x, v, and y are customer visits, swap u and x with v and y;

bool LocalSearch::operateMoveDepotRouteM6(Route& ru, Route& rv, bool equal) {

    bool result = false;
    float costU, costV;
    int demandU, demandV, serviceU, serviceV, customerU, customerUPP, customerV, customerVPP;

    int i, j;

    if (equal) {

        if (ru.getTour().size() < 4)
            return result;

        i = -1;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

            i++;

            if (next(iterRU) == ru.getTour().end())
                continue;

            auto nextIterRU = next(iterRU);

            //U, U+1
            customerU = (*iterRU);
            customerUPP = (*nextIterRU);

            j = i;

            //for (auto iterRV = ru.getTour().begin(); iterRV != ru.getTour().end(); ++iterRV) {
            for (auto iterRV = next(nextIterRU); iterRV != ru.getTour().end(); ++iterRV) {

                j++;

                if (next(iterRV) == ru.getTour().end())
                    continue;

                auto nextIterRV = next(iterRV);

                //V, V+1
                customerV = (*iterRV);
                customerVPP = (*nextIterRV);

                // Avoid repeated customers
                if (customerV == customerU || customerV == customerUPP)
                    continue;

                if (customerVPP == customerU || customerVPP == customerUPP)
                    continue;

                costU = ru.getTotalCost();

                // Change V with U
                ru.changeCustomer(iterRV, customerU);
                ru.changeCustomer(nextIterRV, customerUPP);

                //cout << "Equal: " << equal << " -> i: " << i << " - j: " << j << " \tU: " << customerU << " - X: " << customerUPP
                //    << " \tV: " << customerV << " - Y: " << customerVPP << endl;

                //ru.printSolution();

                // Change U with V
                ru.changeCustomer(iterRU, customerV);
                ru.changeCustomer(nextIterRU, customerVPP);

                //ru.printSolution();

                if (Util::isBetterSolution(ru.getTotalCost(), costU)) {
                    result = true;
                    break;
                } else {
                    // Put U and V in correct place

                    // Put U again
                    ru.changeCustomer(iterRU, customerU);
                    ru.changeCustomer(nextIterRU, customerUPP);

                    // Put V again
                    ru.changeCustomer(iterRV, customerV);
                    ru.changeCustomer(nextIterRV, customerVPP);

                }

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    } else {

        if (ru.getTour().size() < 2 || rv.getTour().size() < 2)
            return result;

        i = -1;

        for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {
            i++;
            if (next(iterRU) == ru.getTour().end())
                continue;

            auto nextIterRU = next(iterRU);

            //U, U+1
            customerU = (*iterRU);
            customerUPP = (*nextIterRU);

            demandU = ru.getProblem()->getDemand().at(customerU - 1);
            demandU += ru.getProblem()->getDemand().at(customerUPP - 1);

            serviceU = ru.getProblem()->getServiceTime().at(customerU - 1);
            serviceU += ru.getProblem()->getServiceTime().at(customerUPP - 1);

            j = -1;
            for (auto iterRV = rv.getTour().begin(); iterRV != rv.getTour().end(); ++iterRV) {

                j++;

                if (next(iterRV) == rv.getTour().end())
                    continue;

                auto nextIterRV = next(iterRV);

                //V, V+1
                customerV = (*iterRV);
                customerVPP = (*nextIterRV);

                demandV = rv.getProblem()->getDemand().at(customerV - 1);
                demandV += rv.getProblem()->getDemand().at(customerVPP - 1);

                serviceV = rv.getProblem()->getServiceTime().at(customerV - 1);
                serviceV += rv.getProblem()->getServiceTime().at(customerVPP - 1);

                if (CHECK_DEMAND) {

                    // If the capacity is exceeded swapping U and V
                    if (ru.getDemand() - demandU + demandV > ru.getCapacity())
                        continue;
                }

                if (CHECK_DEMAND) {

                    // If the capacity is exceeded swapping V and U
                    if (rv.getDemand() - demandV + demandU > rv.getCapacity())
                        continue;

                }

                // Avoid repeated customers
                if (customerV == customerU || customerV == customerUPP)
                    continue;

                if (customerVPP == customerU || customerVPP == customerUPP)
                    continue;

                costU = ru.getTotalCost();
                costV = rv.getTotalCost();

                // Change V with U
                rv.changeCustomer(iterRV, customerU);
                rv.changeCustomer(nextIterRV, customerUPP);

                //rv.printSolution();

                // Change U with V
                ru.changeCustomer(iterRU, customerV);
                ru.changeCustomer(nextIterRU, customerVPP);

                //ru.printSolution();

                //cout << "Equal: " << equal << " -> i: " << i << " - j: " << j << " \tU: " << customerU << " - X: " << customerUPP
                //    << " \tV: " << customerV << " - Y: " << customerVPP << endl;

                if (Util::isBetterSolution(ru.getTotalCost() + rv.getTotalCost(), costU + costV)) {
                    result = true;
                    break;
                } else {
                    // Put U and V in correct place

                    // Put U again
                    ru.changeCustomer(iterRU, customerU);
                    ru.changeCustomer(nextIterRU, customerUPP);

                    // Put V again
                    rv.changeCustomer(iterRV, customerV);
                    rv.changeCustomer(nextIterRV, customerVPP);

                }

                if (ru.getProblem()->getMonitor().isTerminated())
                    break;

            }

            if (result)
                break;

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

    }

    return result;
}

// (M7) If r(u)=r(v),   replace (u,x) and (v,y) by (u,v) and (x,y);

bool LocalSearch::operateMoveDepotRouteM7(Route& ru, Route& rv, bool equal) {

    bool result = false;
    float cost;

    if (!equal)
        return result;

    if (ru.getTour().size() < 4)
        return result;

    for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

        auto iterRX = next(iterRU);

        if (iterRX == ru.getTour().end())
            break;

        for (auto iterRV = next(iterRX); iterRV != ru.getTour().end(); ++iterRV) {

            auto iterRY = next(iterRV);

            if (iterRY == ru.getTour().end())
                break;

            cost = ru.getTotalCost();

            auto iterStart = iterRX;
            auto iterEnd = iterRV;

            // Reverse from RV to RX
            ru.reverse(iterStart, iterEnd);
            //ru.printSolution();

            if (Util::isBetterSolution(ru.getTotalCost(), cost)) {
                result = true;
                break;
            } else {
                // Reverse again from RV to RX
                ru.reverse(iterStart, iterEnd);
            }

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

        if (result)
            break;

        if (ru.getProblem()->getMonitor().isTerminated())
            break;

    }

    return result;

}

// (M8) If r(u) != r(v),replace (u,x) and (v,y) by (u,v) and (x,y);

bool LocalSearch::operateMoveDepotRouteM8(Route& ru, Route& rv, bool equal) {

    bool result = false;
    bool process = false;
    bool stop = false;

    if (equal)
        return result;

    if (ru.getTour().size() < 2)
        return result;

    if (rv.getTour().size() < 2)
        return result;

    for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

        auto iterRX = next(iterRU);

        if (iterRX == ru.getTour().end())
            break;

        for (auto iterRV = rv.getTour().begin(); iterRV != rv.getTour().end(); ++iterRV) {

            auto iterRY = next(iterRV);

            if (iterRY == rv.getTour().end())
                break;

            Route newRU = Route(ru.getProblem(), ru.getConfig(), ru.getDepot(), ru.getId());
            Route newRV = Route(rv.getProblem(), rv.getConfig(), rv.getDepot(), rv.getId());

            // Stop at first penalty -----
            stop = false;

            // From D->U
            for (auto ite = ru.getTour().begin(); ite != iterRX; ++ite) {
                newRU.addAtBack((*ite));

                if (newRU.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;

            // From V->D
            auto ite = iterRV;
            process = true;

            while (process) {

                if (ite == rv.getTour().begin())
                    process = false;

                newRU.addAtBack((*ite));

                if (newRU.isPenalized()) {
                    stop = true;
                    break;
                }

                if (ite != rv.getTour().begin())
                    --ite;
            }

            if (stop)
                continue;

            //V: From D->X
            for (auto ite = prev(ru.getTour().end()); ite != iterRU; --ite) {
                newRV.addAtBack((*ite));

                if (newRV.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;


            //V: From Y->D
            for (auto ite = iterRY; ite != rv.getTour().end(); ++ite) {
                newRV.addAtBack((*ite));

                if (newRV.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;

            //newRU.printSolution();
            //newRV.printSolution();

            if (Util::isBetterSolution(newRU.getTotalCost() + newRV.getTotalCost(),
                    ru.getTotalCost() + rv.getTotalCost())) {
                ru = newRU;
                rv = newRV;
                result = true;
                break;
            }

            if (ru.getProblem()->getMonitor().isTerminated())
                break;


        }

        if (result)
            break;

        if (ru.getProblem()->getMonitor().isTerminated())
            break;

    }

    return result;

}

// (M9) If r(u) != r(v),replace (u,x) and (v,y) by (u,y) and (x,v).

bool LocalSearch::operateMoveDepotRouteM9(Route& ru, Route& rv, bool equal) {

    bool result = false;
    bool process = false;
    bool stop = false;

    if (equal)
        return result;

    if (ru.getTour().size() < 2)
        return result;

    if (rv.getTour().size() < 2)
        return result;

    for (auto iterRU = ru.getTour().begin(); iterRU != ru.getTour().end(); ++iterRU) {

        auto iterRX = next(iterRU);

        if (iterRX == ru.getTour().end())
            break;

        for (auto iterRV = rv.getTour().begin(); iterRV != rv.getTour().end(); ++iterRV) {

            auto iterRY = next(iterRV);

            if (iterRY == rv.getTour().end())
                break;

            Route newRU = Route(ru.getProblem(), ru.getConfig(), ru.getDepot(), ru.getId());
            Route newRV = Route(rv.getProblem(), rv.getConfig(), rv.getDepot(), rv.getId());

            // Stop at first penalty -----
            stop = false;

            //U: From D->U
            for (auto ite = ru.getTour().begin(); ite != iterRX; ++ite) {
                newRU.addAtBack((*ite));

                if (newRU.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;

            //U: From Y->D
            for (auto ite = iterRY; ite != rv.getTour().end(); ++ite) {
                newRU.addAtBack((*ite));

                if (newRU.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;

            //OLD: V: From D->X
            //for (auto ite = prev(ru.getTour().end()); ite != iterRU; --ite) {

            //V: From X->D
            for (auto ite = iterRX; ite != ru.getTour().end(); ++ite) {
                newRV.addAtBack((*ite));

                if (newRV.isPenalized()) {
                    stop = true;
                    break;
                }

            }

            if (stop)
                continue;

            //V: From V->D
            auto ite = iterRV;
            process = true;

            while (process) {

                if (ite == rv.getTour().begin())
                    process = false;

                newRV.addAtBack((*ite));

                if (newRV.isPenalized()) {
                    stop = true;
                    break;
                }

                if (ite != rv.getTour().begin())
                    --ite;
            }

            if (stop)
                continue;

            //cout << "U = " << (*iterRU) << " - X = " << (*iterRX);
            //cout << " / V = " << (*iterRV) << " - Y = " << (*iterRY);
            //cout << endl;

            //newRU.printSolution();
            //newRV.printSolution();

            if (Util::isBetterSolution(newRU.getTotalCost() + newRV.getTotalCost(),
                    ru.getTotalCost() + rv.getTotalCost())) {
                ru = newRU;
                rv = newRV;
                result = true;
                break;
            }

            if (ru.getProblem()->getMonitor().isTerminated())
                break;

        }

        if (result)
            break;

        if (ru.getProblem()->getMonitor().isTerminated())
            break;

    }

    return result;
}
