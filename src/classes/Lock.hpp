/* 
 * File:   Lock.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on September 11, 2014, 12:01 PM
 */

#ifndef LOCK_HPP
#define	LOCK_HPP

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <chrono>
#include <atomic>         // std::atomic

//http://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
//struct mutex_wrapper : std::mutex
//{
//  mutex_wrapper() = default;
//  mutex_wrapper(mutex_wrapper const&) noexcept : std::mutex() {}
//  bool operator==(mutex_wrapper const&other) noexcept { return this==&other; }
//};
//
//struct condvar_wrapper : std::condition_variable
//{
//  condvar_wrapper() = default;
//  condvar_wrapper(condvar_wrapper const&) noexcept : std::condition_variable() {}
//  bool operator==(condvar_wrapper const&other) noexcept { return this==&other; }
//};

class Lock {
    
    std::mutex mutexLocker;
    //mutex_wrapper mutexLocker;
    std::condition_variable conditionVariable;
    //condvar_wrapper conditionVariable;
    
    int id;
    
    std::atomic<bool> ready;// (false);    
    bool terminated;

public:
    
    Lock();
    virtual ~Lock();

    int getId() const;
    void setId(int id);
    
    bool isReady() const;
    void setReady(bool ready);
    
    bool isTerminated() const;
    void setTerminated(bool terminated);
    
    void lock();
    void unlock();
    
    void wait(bool readyState);
    void notify(bool ready);
    
private:

    std::condition_variable& getConditionVariable();    
    //condvar_wrapper& getConditionVariable();
    std::mutex& getMutexLocker();
    //mutex_wrapper& getMutexLocker();
    
};

#endif	/* LOCK_HPP */

