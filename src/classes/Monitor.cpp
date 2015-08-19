/* 
 * File:   Monitor.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on August 14, 2014, 4:03 PM
 */

#include "Monitor.hpp"

/*
 * Constructors and Destructor
 */

Monitor::Monitor() {
    this->createLockIDs();
}

Monitor::Monitor(const Monitor& orig) {
}

Monitor::~Monitor() {

}

/*
 * Getters and Setters
 */

bool Monitor::isStarted() const {
    return started;
}

void Monitor::setStarted(bool started) {
    this->started = started;
}

unsigned long int Monitor::getGeneration() const {
    return generation;
}

void Monitor::setGeneration(unsigned long int generation) {
    this->generation = generation;
}

time_t Monitor::getStart() const {
    return start;
}

void Monitor::setStart(time_t start) {
    this->start = start;
}

time_t Monitor::getLastTimeUpdate() const {
    return lastTimeUpdate;
}

void Monitor::setLastTimeUpdate(time_t lastTimeUpdate) {
    this->lastTimeUpdate = lastTimeUpdate;
}

bool Monitor::isEvolvingSubpops() const {
    return evolvingSubpops;
}

void Monitor::setEvolvingSubpops(bool evolvingSubpops) {
    this->evolvingSubpops = evolvingSubpops;
}

bool Monitor::isEvaluatingSolutions() const {
    return evaluatingSolutions;
}

void Monitor::setEvaluatingSolutions(bool evaluatingSolutions) {
    this->evaluatingSolutions = evaluatingSolutions;
}

bool Monitor::isUpdatingBestInds() const {
    return updatingBestInds;
}

void Monitor::setUpdatingBestInds(bool updatingBest) {
    this->updatingBestInds = updatingBest;
}

bool Monitor::isEliteGroupLocalSearch() const {
    return eliteGroupLocalSearch;
}

void Monitor::setEliteGroupLocalSearch(bool eliteGroupLocalSearch) {
    this->eliteGroupLocalSearch = eliteGroupLocalSearch;
}

bool Monitor::isUpdatingEliteGroup() const {
    return updatingEliteGroup;
}

void Monitor::setUpdatingEliteGroup(bool updatingEliteGroup) {
    this->updatingEliteGroup = updatingEliteGroup;
}

bool Monitor::isTerminated() const {
    return terminated;
}

void Monitor::setTerminated(bool terminated) {
    this->terminated = terminated;

    if (this->isTerminated())
        this->terminateLocks();

}

std::mutex& Monitor::getMutexLocker() {
    return mutexLocker;
}

bool Monitor::isDoUpdateBest() const {
    return doUpdateBest;
}

void Monitor::setDoUpdateBest(bool doUpdateBest) {
    this->doUpdateBest = doUpdateBest;
}

bool Monitor::isCopyFromSubpopsRequested() const {
    return copyFromSubpopsRequested;
}

void Monitor::setCopyFromSubpopsRequested(bool copyFromSubpopsRequested) {
    this->copyFromSubpopsRequested = copyFromSubpopsRequested;
}

bool Monitor::isCopyFromSubpopsAllowed() const {
    return copyFromSubpopsAllowed;
}

void Monitor::setCopyFromSubpopsAllowed(bool copyFromSubpopsAllowed) {
    this->copyFromSubpopsAllowed = copyFromSubpopsAllowed;
}

bool Monitor::isCopyBestIndsRequested() const {
    return copyBestIndsRequested;
}

void Monitor::setCopyBestIndsRequested(bool copyBestIndsRequested) {
    this->copyBestIndsRequested = copyBestIndsRequested;
}

bool Monitor::isCopyBestIndsAllowed() const {
    return copyBestIndsAllowed;
}

void Monitor::setCopyBestIndsAllowed(bool copyBestIndsAllowed) {
    this->copyBestIndsAllowed = copyBestIndsAllowed;
}

bool Monitor::isUpdateBestIndsRequested() const {
    return updateBestIndsRequested;
}

void Monitor::setUpdateBestIndsRequested(bool updateEGRequested) {
    this->updateBestIndsRequested = updateEGRequested;
}

bool Monitor::isUpdateBestIndsAllowed() const {
    return updateBestIndsAllowed;
}

void Monitor::setUpdateBestIndsAllowed(bool updateEGAllowed) {
    this->updateBestIndsAllowed = updateEGAllowed;
}

bool Monitor::isUpdateEGRequested() const {
    return updateEGRequested;
}

void Monitor::setUpdateEGRequested(bool updateEGRequested) {
    this->updateEGRequested = updateEGRequested;
}

Lock* Monitor::getLock(int subpopId, int indId) {
    //printf("subpopId: %d \tindId: %d\n", subpopId, indId);
    return &locks[subpopId][indId];
    //return locks.at(subpopId).at(indId);
}

Lock* Monitor::getSubpopLock(int subpopId) {
    return &subpopLocks[subpopId];
}


/*
 * Public Methods
 */

void Monitor::updateGeneration() {
    this->setGeneration(this->getGeneration() + 1);
}

void Monitor::createLocks(int numSubpop, int numId) {

    //    locks = new Lock*[numSubpop];
    //
    //    for (int i = 0; i < numSubpop; ++i)
    //        locks[i] = new Lock[numId];

    //    locksWrapper.resize(numSubpop * numId);

    //    locks = typedef_vectorMatrix<Lock>(numSubpop);
    //    for (int i = 0; i < numSubpop; ++i)
    //        locks.at(i).resize(numId);

}

void Monitor::destroyLocks(int numSubpop) {

    //    for (int i = 0; i < numSubpop; ++i)
    //        delete [] locks[i];
    //
    //    delete [] locks;

}

bool Monitor::forceLock() {

    bool lock = this->getMutexLocker().try_lock();

    int n, nMax = 100;
    n = 0;

//    printf("try_lock();start\n");
    while (!lock && n < nMax) {

        lock = this->getMutexLocker().try_lock();
        n++;
        //printf("try_lock():waiting;\n");

        if (lock)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(3));

        if (this->isTerminated())
            break;
    }

/*    if(!lock)
        printf("try_lock();ERROR\n");    */    
    
    return lock;

}

void Monitor::addToLog(string text) {

    typedef_log l;
    l.time = Util::diffTimeFromStart(this->getStart());
    l.text = text;

    //this->getLog().push_back(l);
    printf("%.2f: %s\n", l.time, l.text.c_str());

}

void Monitor::printLog() {

    for (typedef_log& l : this->getLog()) {
        printf("%.2f: %s\n", l.time, l.text.c_str());
    }

}

/*
 * Private Methods
 */

vector<typedef_log>& Monitor::getLog() {
    return log;
}

void Monitor::terminateLocks() {

    for (int d = 0; d < NUM_MAX_DEP; ++d) {
        this->getSubpopLock(d)->setTerminated(true);
        
        for (int c = 0; c < NUM_MAX_CLI; ++c)
            this->getLock(d, c)->setTerminated(true);
    }

}

void Monitor::createLockIDs() {

    for (int d = 0; d < NUM_MAX_DEP; ++d) {
        this->getSubpopLock(d)->setId(d);
        
        for (int c = 0; c < NUM_MAX_CLI; ++c)
            this->getLock(d, c)->setId((d + 1) * 100 + c);
    }
    
}
