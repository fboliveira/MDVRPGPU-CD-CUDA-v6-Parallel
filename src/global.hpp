/*
 * File:   global.h
 * Author: fernando
 *
 * Created on May 15, 2013, 9:30 PM
 */

#include <vector>
#include <list>
#include <iterator>
#include <string>
#include <time.h>

using namespace std;

#ifndef GLOBAL_H

#define	GLOBAL_H

// INSTANCE TEST
#define INST_TEST "p24"

#define SOURCE 4
#define DEBUG_VERSION false

#if SOURCE==1
// LOCAL
#define BASE_DIR_DAT "/Users/fernando/Temp/MDVRP/dat/"
#define BASE_DIR_SOL "/Users/fernando/Temp/MDVRP/sol/"
//#define LOG_RUN_FILE "/Users/fernando/Temp/MDVRP/experiments/mdvrpcpu-teste.txt"
#define LOG_RUN_FILE "/Users/fernando/Temp/MDVRP/mdvrpcpu-20.txt"

#elif SOURCE==2

// UFMG
#define BASE_DIR_DAT "/home/fernando/experiments/instances/dat/"
#define BASE_DIR_SOL "/home/fernando/experiments/instances/sol/"
#define LOG_RUN_FILE "/home/fernando/experiments/mdvrpcpu-99-01-03.txt"

#elif SOURCE==3

// HUGO
#define BASE_DIR_DAT "/home/fernando/Temp/MDVRP/dat/"
#define BASE_DIR_SOL "/home/fernando/Temp/MDVRP/sol/"
#define LOG_RUN_FILE "/home/fernando/Temp/MDVRP/experiments/mdvrpcpu-99-01-02.txt"

#elif SOURCE==4

// HUGO Win
#define BASE_DIR_DAT "C:/Dev/MDVRP/dat/"
#define BASE_DIR_SOL "C:/Dev/MDVRP/sol/"
#define LOG_RUN_FILE "C:/Dev/MDVRP/mdvrpGPU-hugo-01-01.txt"

#endif

#define DEPOT_DELIM -1
#define ROUTE_DELIM -2

#define MIN_ELEM_IND 5

#define NUM_MAX_DEP 9
#define NUM_MAX_CLI 360

enum class Enum_Algorithms {
    SSGA,
    SSGPU,
    ES
};

enum Enum_StopCriteria
{
    NUM_GER,
    TEMPO
};

enum Enum_Process_Type
{
    MONO_THREAD,
    MULTI_THREAD,
    GPU_VERSION
};

enum Enum_Local_Search_Type
{
    RANDOM,
    SEQUENTIAL,
    NOT_APPLIED
};

template<class T>
using typedef_vectorMatrix = vector<vector<T>>;

using typedef_vectorIntIterator = vector<int>::iterator;
using typedef_vectorIntSize = vector<int>::size_type;
using typedef_listIntIterator = list<int>::iterator;

typedef struct {
    int i; // id, indice, ...
    float x;
    float y;
} typedef_point;

typedef struct {
    int index;
    float cost;
} typedef_order;

typedef struct {
    double time;
    float cost;
} typedef_evolution;

typedef struct {
    double time;
    string text;
} typedef_log;

typedef struct {
    int depot;
    typedef_vectorIntIterator position;
    float cost;
} typedef_location;

template<typename T> struct matrix_special
{
    T *data;
    int lines;
    int columns;

};

//--template matrix_special<float>;

#endif	/* GLOBAL_H */
