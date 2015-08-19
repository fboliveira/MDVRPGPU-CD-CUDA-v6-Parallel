/*
 * File:   Config.cpp
 * Author: fernando
 *
 * Created on July 21, 2014, 4:22 PM
 */

#include "AlgorithmConfig.hpp"

/*
 * Constructors
 */

AlgorithmConfig::AlgorithmConfig() {

}

/*
 * Getters and Setters
 */

Enum_Algorithms AlgorithmConfig::getAlgorithm() const {
    return algorithm;
}

void AlgorithmConfig::setAlgorithm(Enum_Algorithms algorithm) {
    this->algorithm = algorithm;
}

float AlgorithmConfig::getCapacityPenalty() const {
    return capacityPenalty;
}

void AlgorithmConfig::setCapacityPenalty(float capacityPenalty) {
    this->capacityPenalty = capacityPenalty;
}

bool AlgorithmConfig::isDebug() const {
    return debug;
}

void AlgorithmConfig::setDebug(bool debug) {
    this->debug = debug;
}

bool AlgorithmConfig::isDisplay() const {
    return display;
}

void AlgorithmConfig::setDisplay(bool display) {
    this->display = display;
}

float AlgorithmConfig::getExecutionTime() const {
    return executionTime;
}

void AlgorithmConfig::setExecutionTime(float executionTime) {
    this->executionTime = executionTime;
}

float AlgorithmConfig::getMaxTimeWithoutUpdate() const {
    return maxTimeWithoutUpdate;
}

void AlgorithmConfig::setMaxTimeWithoutUpdate(float maxTimeWithoutUpdate) {
    this->maxTimeWithoutUpdate = maxTimeWithoutUpdate;
}

float AlgorithmConfig::getExtraVehiclesPenalty() const {
    return extraVehiclesPenalty;
}

void AlgorithmConfig::setExtraVehiclesPenalty(float extraVehiclesPenalty) {
    this->extraVehiclesPenalty = extraVehiclesPenalty;
}

float AlgorithmConfig::getIncompleteSolutionPenalty() const {
    return incompleteSolutionPenalty;
}

void AlgorithmConfig::setIncompleteSolutionPenalty(float incompleteSolutionPenalty) {
    this->incompleteSolutionPenalty = incompleteSolutionPenalty;
}

Enum_Local_Search_Type AlgorithmConfig::getLocalSearchType() const {
    return localSearchType;
}

void AlgorithmConfig::setLocalSearchType(Enum_Local_Search_Type localSearchType) {
    this->localSearchType = localSearchType;
}

float AlgorithmConfig::getMutationRatePLS() const {
    return mutationRatePLS;
}

void AlgorithmConfig::setMutationRatePLS(float mutationRatePLS) {
    this->mutationRatePLS = mutationRatePLS;
}

float AlgorithmConfig::getMutationRatePM() const {
    return mutationRatePM;
}

void AlgorithmConfig::setMutationRatePM(float mutationRatePM) {
    this->mutationRatePM = mutationRatePM;
}

unsigned long int AlgorithmConfig::getNumGen() const {
    return numGen;
}

void AlgorithmConfig::setNumGen(unsigned long int numGen) {
    this->numGen = numGen;
}

int AlgorithmConfig::getNumSolutionElem() const {
    return numSolutionElem;
}

void AlgorithmConfig::setNumSolutionElem(int numSolutionElem) {
    this->numSolutionElem = numSolutionElem;
}

int AlgorithmConfig::getNumSubIndDepots() const {
    return numSubIndDepots;
}

void AlgorithmConfig::setNumSubIndDepots(int numSubIndDepots) {
    this->numSubIndDepots = numSubIndDepots;
}

int AlgorithmConfig::getNumOffspringsPerParent() const {
    return numOffspringsPerParent;
}

void AlgorithmConfig::setNumOffspringsPerParent(int numOffspringsPerParent) {
    this->numOffspringsPerParent = numOffspringsPerParent;
}

Enum_Process_Type AlgorithmConfig::getProcessType() const {
    return processType;
}

void AlgorithmConfig::setProcessType(Enum_Process_Type processType) {
    this->processType = processType;
}

float AlgorithmConfig::getRouteDurationPenalty() const {
    return routeDurationPenalty;
}

void AlgorithmConfig::setRouteDurationPenalty(float routeDurationPenalty) {
    this->routeDurationPenalty = routeDurationPenalty;
}

bool AlgorithmConfig::isSaveLogRunFile() const {
    return saveLogRunFile;
}

void AlgorithmConfig::setSaveLogRunFile(bool saveLogRunFile) {
    this->saveLogRunFile = saveLogRunFile;
}

Enum_StopCriteria AlgorithmConfig::getStopCriteria() const {
    return stopCriteria;
}

void AlgorithmConfig::setStopCriteria(Enum_StopCriteria stopCriteria) {
    this->stopCriteria = stopCriteria;
}

int AlgorithmConfig::getTotalMoves() const {
    return totalMoves;
}

void AlgorithmConfig::setTotalMoves(int totalMoves) {
    this->totalMoves = totalMoves;
}

bool AlgorithmConfig::isWriteFactors() const {
    return writeFactors;
}

void AlgorithmConfig::setWriteFactors(bool writeFactors) {
    this->writeFactors = writeFactors;
}

int AlgorithmConfig::getEliteGroupLimit() const {
    return this->eliteGroupLimit;
}

void AlgorithmConfig::setEliteGroupLimit(int eliteGroupLimit) {
    this->eliteGroupLimit = eliteGroupLimit;
}

/*
 * Methods
 */

void AlgorithmConfig::setParameters(MDVRPProblem *problem) {

    // Algoritmo a ser executado
    this->setAlgorithm(Enum_Algorithms::ES);

    // Criterio de parada utilizado
    this->setStopCriteria(TEMPO);
    //this->setStopCriteria(NUM_GER); 

    // Tipo de processamento
    this->setProcessType(MULTI_THREAD);
    //this->setProcessType(MONO_THREAD);

    // MODO DE DEBUG
    this->setDebug(DEBUG_VERSION);

    // Exibir ou nao informacoes durante o processo
    this->setDisplay(true);

    if (isDebug()) { // Debug version
        // Tempo limite de execucao (s) para criterioParada == TEMPO
        this->setExecutionTime(10 * 60.0); // 1.800 seg

        // Max time without update
        this->setMaxTimeWithoutUpdate(5 * 60.0); // 600 seg.

        // Numero de individuos da subpopulacao de depositos
        // Mu value
        this->setNumSubIndDepots(3);
        // Lambda value
        this->setNumOffspringsPerParent(1);

        // Elite group size
        this->setEliteGroupLimit(5);
    }
    else { // Production version
        // Tempo limite de execucao (s) para criterioParada == TEMPO
        this->setExecutionTime(30 * 60.0); // 1.800 seg

        // Max time without update
        this->setMaxTimeWithoutUpdate(10 * 60.0); // 600 seg.

        // Numero de individuos da subpopulacao de depositos
        // Mu value
        this->setNumSubIndDepots(20);
        // Lambda value
        this->setNumOffspringsPerParent(5);

        // Elite group size
        this->setEliteGroupLimit(50);
    }

#if SOURCE==1
    // Save log run to file -- last result
    this->setSaveLogRunFile(false);
#else
    // Save log run to file -- last result
    this->setSaveLogRunFile(true);
#endif

    // Numero de elementos na subpopulacao de solucoes
    // n - problema.depositos
    // 1 - FO
    this->setNumSolutionElem(problem->getDepots() + 1);

    // PARAMETROS PARA A EVOLUCAO -------
    // - Taxa de Mutacao
    // -- Aleatoria
    this->setMutationRatePM(1.0);
    // -- Busca local
    this->setLocalSearchType(Enum_Local_Search_Type::RANDOM);
    this->setMutationRatePLS(0.2);
    // ----------------------------

    // Penalidade para a restricao de capacidade dos veiculos
    this->setCapacityPenalty(pow(10, 3));

    // Extra vehicles
    // -- Penalty
    this->setExtraVehiclesPenalty(pow(10, 3));
    // -- Relaxed version
    //this->setExtraVehiclesPenalty(0.0);

    // Route duration
    if (problem->getDuration() > 0)
        this->setRouteDurationPenalty(pow(10, 3));
    else
        this->setRouteDurationPenalty(0.0);

    // Penalidade para solucoes incompletas
    this->setIncompleteSolutionPenalty(pow(10, 5));

    // Total de movimentos - LS
    this->setTotalMoves(9);

    this->setWriteFactors(false);

}

string AlgorithmConfig::getLocalSearchTypeStringValue() {

    string ls;

    switch (this->getLocalSearchType()) {

    case Enum_Local_Search_Type::RANDOM:
        ls = "RND";
        break;

    case Enum_Local_Search_Type::SEQUENTIAL:
        ls = "SQT";
        break;

    case Enum_Local_Search_Type::NOT_APPLIED:
        ls = "NTA";
        break;

    }

    return ls;

}