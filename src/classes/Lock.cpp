/* 
 * File:   Lock.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on September 11, 2014, 12:01 PM
 */

#include "Lock.hpp"
#include "MDVRPProblem.hpp"

/*
 * Constructors and Destructor
 */

Lock::Lock() {
    this->setReady(false);
    this->setTerminated(false);
}

Lock::~Lock() {
    //this->unlock();
}
/*
 * Getters and Setters
 */

int Lock::getId() const {
    return id;
}

void Lock::setId(int id) {
    this->id = id;
}

bool Lock::isReady() const {
    return ready;
}

void Lock::setReady(bool ready) {
    this->ready = ready;
}

bool Lock::isTerminated() const {
    return terminated;
}

void Lock::setTerminated(bool terminated) {
    this->terminated = terminated;
}

/*
 * Public Methods
 */

void Lock::lock() {
    this->getMutexLocker().lock();
}

void Lock::unlock() {
    this->getMutexLocker().unlock();
}

// http://www.cplusplus.com/reference/condition_variable/condition_variable/

void Lock::wait(bool readyState) {

    //printf("Lock::wait::start => %d -- %d - %d\n", this->getId(), this->isReady(), readyState);
    
    std::unique_lock<std::mutex> lck(this->getMutexLocker());
    while (readyState != this->isReady()) {
        this->getConditionVariable().wait_for(lck, std::chrono::milliseconds(3));
        //this->getConditionVariable().wait(lck);

        if (this->isTerminated())
            break;
    }

    //printf("Lock::wait::end => %d -- %d - %d\n", this->getId(), this->isReady(), readyState);

}

void Lock::notify(bool ready) {
    //printf("Lock::notify::start => %d -- %d - %d\n", this->getId(), this->isReady(), ready);
    std::unique_lock<std::mutex> lck(this->getMutexLocker());
    this->setReady(ready);
    this->getConditionVariable().notify_all();
    //printf("Lock::notify::end => %d -- %d - %d\n", this->getId(), this->isReady(), ready);
}

/*
 * Private Methods
 */

std::condition_variable& Lock::getConditionVariable() {
    //condvar_wrapper& Lock::getConditionVariable() {    
    return conditionVariable;
}

std::mutex& Lock::getMutexLocker() {
    //mutex_wrapper& Lock::getMutexLocker() {
    return mutexLocker;
}