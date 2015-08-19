/* 
 * File:   Monitor.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on August 14, 2014, 4:03 PM
 */

#ifndef MONITOR_HPP
#define	MONITOR_HPP

#include <string>
#include <mutex>
#include <thread>         // std::thread
#include <chrono>         // std::chrono::seconds

#include "Util.hpp"
#include "Lock.hpp"
#include "../global.hpp"

using namespace std;

class Monitor {
    
    bool started = false;
    
    bool evolvingSubpops = false;
    bool evaluatingSolutions = false;
    bool updatingBestInds = false;
    bool eliteGroupLocalSearch = false;
    bool updatingEliteGroup = false;
    
    bool doUpdateBest = false;
    
    bool copyFromSubpopsRequested = false;
    bool copyFromSubpopsAllowed = false;
    
    bool copyBestIndsRequested = false;
    bool copyBestIndsAllowed = false;
    
    bool updateBestIndsRequested = false;
    bool updateBestIndsAllowed = false;    

    bool updateEGRequested = false;
    
    time_t start;
    time_t lastTimeUpdate;
    
    unsigned long int generation = 0;
    vector<typedef_log> log;

    bool terminated = false; // Indicate if the process is terminated
    std::mutex mutexLocker;
    //std::recursive_mutex mutexLocker;

    //Lock **locks;
    Lock locks[NUM_MAX_DEP][NUM_MAX_CLI];
    Lock subpopLocks[NUM_MAX_DEP];
    
    //std::vector<mutex_wrapper> locksWrapper;
    //typedef_vectorMatrix<mutex_wrapper> locksWrapper;
    //typedef_vectorMatrix<Lock> locks;
        
public:

    Monitor();
    Monitor(const Monitor& orig);
    virtual ~Monitor();

    /*
     * Getters and Setters
     */

    bool isStarted() const;
    void setStarted(bool started);

    unsigned long int getGeneration() const;
    void setGeneration(unsigned long int generation);

    time_t getStart() const;
    void setStart(time_t start);

    time_t getLastTimeUpdate() const;
    void setLastTimeUpdate(time_t lastTimeUpdate);
    
    bool isEvolvingSubpops() const;
    void setEvolvingSubpops(bool evolvingSubpops);

    bool isEvaluatingSolutions() const;
    void setEvaluatingSolutions(bool evaluatingSolutions);

    bool isUpdatingBestInds() const;
    void setUpdatingBestInds(bool updatingBest);

    bool isEliteGroupLocalSearch() const;
    void setEliteGroupLocalSearch(bool eliteGroupLocalSearch);

    bool isUpdatingEliteGroup() const;
    void setUpdatingEliteGroup(bool updatingEliteGroup);
    
    bool isTerminated() const;
    void setTerminated(bool terminated);

    std::mutex& getMutexLocker();
    
    bool isDoUpdateBest() const;
    void setDoUpdateBest(bool doUpdateBest);
    
    bool isCopyFromSubpopsRequested() const;
    void setCopyFromSubpopsRequested(bool copyFromSubpopsRequested);

    bool isCopyFromSubpopsAllowed() const;
    void setCopyFromSubpopsAllowed(bool copyFromSubpopsAllowed);
    
    bool isCopyBestIndsAllowed() const;
    void setCopyBestIndsAllowed(bool copyBestIndsAllowed);

    bool isCopyBestIndsRequested() const;
    void setCopyBestIndsRequested(bool copyBestIndsRequested);

    bool isUpdateBestIndsAllowed() const;
    void setUpdateBestIndsAllowed(bool updateEGAllowed);

    bool isUpdateBestIndsRequested() const;
    void setUpdateBestIndsRequested(bool updateEGRequested);    
    
    bool isUpdateEGRequested() const;
    void setUpdateEGRequested(bool updateEGRequested);

    Lock* getLock(int subpopId, int indId);
    Lock* getSubpopLock(int subpopId);
    
    //Lock& getLock(int subpopId, int indId);
    
    /*
     * Public Methods
     */

    void updateGeneration();

    void createLocks(int numSubpop, int numId);
    void destroyLocks(int numSubpop);
    
    bool forceLock();
    
    void addToLog(string text);
    void printLog();

private:

    vector<typedef_log>& getLog();
    void terminateLocks();
    void createLockIDs();
       
};

#endif	/* MONITOR_HPP */

