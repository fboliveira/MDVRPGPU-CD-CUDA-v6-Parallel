/* 
 * File:   Config.hpp
 * Author: fernando
 *
 * Created on July 21, 2014, 4:22 PM
 */

#ifndef CONFIG_HPP
#define	CONFIG_HPP

#include <cmath>
#include <climits>
#include <string>
#include <mutex>

#include "MDVRPProblem.hpp"
#include "../global.hpp"

using namespace std;

class AlgorithmConfig {
    
private:

    // Algoritmo a ser executado        
    Enum_Algorithms algorithm;

    enum Enum_StopCriteria stopCriteria; // Criterio de parada utilizado    
    enum Enum_Process_Type processType; // Tipo de processamento

    bool debug;
    bool display;
       
    unsigned long int numGen;
    float executionTime; // Tempo limite de execucao (s) para criterioParada == TEMPO
    float maxTimeWithoutUpdate;
    
    int totalMoves;
    bool saveLogRunFile;

    int numSubIndDepots; // Mu value
    int numOffspringsPerParent; // Lambda value
    int numSolutionElem;
    
    float mutationRatePM;

    enum Enum_Local_Search_Type localSearchType;
    float mutationRatePLS;

    float capacityPenalty;
    float routeDurationPenalty;
    float extraVehiclesPenalty;
    float incompleteSolutionPenalty;

    bool writeFactors;

    int eliteGroupLimit;

public:

    AlgorithmConfig();

    Enum_Algorithms getAlgorithm() const;
    void setAlgorithm(Enum_Algorithms algorithm);

    float getCapacityPenalty() const;
    void setCapacityPenalty(float capacityPenalty);

    bool isDebug() const;
    void setDebug(bool debug);

    bool isDisplay() const;
    void setDisplay(bool display);

    float getExecutionTime() const;
    void setExecutionTime(float executionTime);

    float getMaxTimeWithoutUpdate() const;
    void setMaxTimeWithoutUpdate(float maxTimeWithoutUpdate);
    
    float getExtraVehiclesPenalty() const;
    void setExtraVehiclesPenalty(float extraVehiclesPenalty);

    float getIncompleteSolutionPenalty() const;
    void setIncompleteSolutionPenalty(float incompleteSolutionPenalty);

    Enum_Local_Search_Type getLocalSearchType() const;
    void setLocalSearchType(Enum_Local_Search_Type localSearchType);

    float getMutationRatePLS() const;
    void setMutationRatePLS(float mutationRatePLS);

    float getMutationRatePM() const;
    void setMutationRatePM(float mutationRatePM);

    unsigned long int getNumGen() const;
    void setNumGen(unsigned long int numGen);

    int getNumSolutionElem() const;
    void setNumSolutionElem(int numSolutionElem);

    int getNumSubIndDepots() const;
    void setNumSubIndDepots(int numSubIndDepots);

    int getNumOffspringsPerParent() const;
    void setNumOffspringsPerParent(int numOffspringsPerParent);
    
    Enum_Process_Type getProcessType() const;
    void setProcessType(Enum_Process_Type processType);

    float getRouteDurationPenalty() const;
    void setRouteDurationPenalty(float routeDurationPenalty);

    bool isSaveLogRunFile() const;
    void setSaveLogRunFile(bool saveLogRunFile);

    Enum_StopCriteria getStopCriteria() const;
    void setStopCriteria(Enum_StopCriteria stopCriteria);

    int getTotalMoves() const;
    void setTotalMoves(int totalMoves);

    bool isWriteFactors() const;
    void setWriteFactors(bool writeFactors);

    int getEliteGroupLimit() const;
    void setEliteGroupLimit(int eliteGroupLimit);
    
    void setParameters(MDVRPProblem *problem);
    string getLocalSearchTypeStringValue();
    
};

#endif	/* CONFIG_HPP */

