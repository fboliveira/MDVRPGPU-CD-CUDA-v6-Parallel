/* 
 * File:   main.cpp
 * Author: fernando
 *
 * Created on May 15, 2014, 6:30 PM
 */

#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <float.h>
#include <iostream>

#include "classes/MDVRPProblem.hpp"
#include "classes/AlgorithmConfig.hpp"
#include "classes/Random.hpp"
#include "classes/ESCoevolMDVRP.hpp"
#include "classes/LocalSearch.hpp"

using namespace std;

int main(int argc, char** argv) {

    MDVRPProblem *problem = new MDVRPProblem();

    time_t start, end;

    char inst[10], dat[300], sol[300];

    // FATORES A SEREM AVALIADOS
    bool change = false;

    int popSize;
    float executionTime;
    float maxTimeWithoutUpdate;
    float mutationRatePLS;
    int eliteGroupLimit;

    strcpy(dat, BASE_DIR_DAT);
    strcpy(sol, BASE_DIR_SOL);

    if (argc == 1) {
        strcat(dat, INST_TEST);
        strcat(sol, INST_TEST);
        strcpy(inst, INST_TEST);
    } else {
        strcat(dat, argv[1]);
        strcat(sol, argv[1]);
        strcpy(inst, argv[1]);

        if (argc > 2) {

            change = true;

            popSize = atoi(argv[2]);
            executionTime = strtof(argv[3], NULL);
            maxTimeWithoutUpdate = strtof(argv[4], NULL);
            mutationRatePLS = strtof(argv[5], NULL);
            eliteGroupLimit = atoi(argv[6]);

        }
    }

    strcat(sol, ".res");

    time(&start);

    /* initialize random seed: */
    Random::randomize();

    // Read data file
    if (!problem->processInstanceFiles(dat, sol, inst)) {
        //std::cout << "Press enter to continue ...";
        //std::cin.get();
        return EXIT_FAILURE;
    }
    
    // Geracao de dados de alocacao - allocation N/M
    //problem->printAllocation();
    problem->printAllocationDependecy();
    return 0;

    // # Configuracoes para execucao
    AlgorithmConfig *config = new AlgorithmConfig();
    config->setParameters(problem);

    if (change) {

        config->setNumSubIndDepots(popSize);
        config->setExecutionTime(executionTime);
        config->setMaxTimeWithoutUpdate(maxTimeWithoutUpdate);
        config->setMutationRatePLS(mutationRatePLS);
        config->setEliteGroupLimit(eliteGroupLimit);

        config->setWriteFactors(true);
    }

    //printf("INSTANCIA...: %s\n", problem->getInstCode().c_str());
    //problem->print();
    //LocalSearch::testFunction(problem, config);
    //return(0);


    printf("=======================================================================================\n");
    printf("CoES -- A Cooperative Coevolutionary Algorithm for Multi-Depot Vehicle Routing Problems\n");

    if (config->getProcessType() == Enum_Process_Type::MONO_THREAD)
        printf("* MONO");
    else
        printf("*** MULTI");

    printf(" Thread version\n");
    printf("=======================================================================================\n\n");

    if (config->isDebug()) {
        printf("D\tE\tB\tU\tG\tversion!\n\n");
        printf("=======================================================================================\n\n");
    }

    printf("INSTANCIA...: %s\n", problem->getInstance().c_str());
    printf("Depositos...: %d\n", problem->getDepots());
    printf("Veiculos....: %d\n", problem->getVehicles());
    printf("Capacidade..: %d\n", problem->getCapacity());
    printf("Duracao.....: %.2f\n", problem->getDuration());
    printf("Clientes....: %d\n", problem->getCustomers());

    if (config->getStopCriteria() == NUM_GER)
        printf("Geracoes....: %lu\n", config->getNumGen());
    else {

        if (!change) {
            printf("Tempo max...: %.2f seg. ", config->getExecutionTime());

            if (config->getMaxTimeWithoutUpdate() > 0 && config->getMaxTimeWithoutUpdate() <= config->getExecutionTime())
                printf("(ou %.2f seg. sem atualizacao)", config->getMaxTimeWithoutUpdate());

            printf("\n");
        }
    }

    printf("\n====Fatores para avaliacao====\n");
    printf("Tam Subpop..: %d\n", config->getNumSubIndDepots());
    //printf("Lambda......: %d\n", config->getNumOffspringsPerParent());

    printf("Tempo max...: %.2f seg. ", config->getExecutionTime());
    printf("(ou %.2f seg. sem atualizacao)\n", config->getMaxTimeWithoutUpdate());

    printf("PLS Rate....: %.2f\n", config->getMutationRatePLS());
    printf("Elite Size..: %d\n", config->getEliteGroupLimit());
    printf("==============================\n\n");

    printf("Busca Local.: %s\n", config->getLocalSearchType() == RANDOM ? "BLA" : "BLS");
    cout << "\n\n";

    ESCoevolMDVRP esCoevolMDVRP = ESCoevolMDVRP(problem, config);
    esCoevolMDVRP.run();

    time(&end);

    double dif = difftime(end, start);
    printf("\n\nProcesso finalizado. Tempo gasto = %lf\n\n", dif);
    printf("Encerramento: ");
    Util::printTimeNow();

    delete problem;
    delete config;

    return EXIT_SUCCESS;
}