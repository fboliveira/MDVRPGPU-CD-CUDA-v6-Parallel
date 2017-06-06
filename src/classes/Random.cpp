/* 
 * File:   Random.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 22, 2014, 9:23 PM
 */

#include "Random.hpp"

/*
 * Constructors
 */

Random::Random() {
}

/*
 * Getters and Setters
 */

/*
 * Methods
 */

void Random::randomize() {
    srand(std::time(0));
}

double Random::randDoubleBetween(double x0, double x1) {
    // Define os numeros aleatorios conforme limite inferior e superior
    // limiteInferior + aleatorios * (limiteSuperior - limiteInferior);
    //return x0 + (x1 - x0) * randfloat(); // / ((double) RAND_MAX);

    //return (randfloat() % x1) + x0;
    Util::error("TO-DO: randdoublebet", 0);
    return -1.0;

}

float Random::randFloat() {
    return (float) rand() / (float) RAND_MAX;
}

int Random::randInt() {
    return randIntBetween(0, RAND_MAX);
}

// Numero inteiro aleatorio [x0..x1]

int Random::randIntBetween(int x0, int x1) {
    // Define os numeros aleatorios conforme limite inferior e superior
    // limiteInferior + aleatorios * (limiteSuperior - limiteInferior);

    // http://stackoverflow.com/questions/4413170/c-generating-a-truly-random-number-between-10-20

    if (x0 > x1) {
        Util::error("randIntBetween: x0 > x1 ", -1);
        cout << "x0: " << x0 << "\tx1: " << x1 << "\n";
        return -1;
    }

    return (std::rand() % (x1 + 1 - x0)) +x0;

//    return uniformIntDiscreteDistribution(x0, x1);
    
}

void Random::randPermutation(int size, int min, vector<int>& elements) {

    int i;

    // inizialize
    for (i = 0; i < size; ++i) {
        elements.push_back(min);
        min++;
    }

    random_shuffle(elements.begin(), elements.end());

}

void Random::randTwoNumbers(int min, int max, int& n1, int& n2) {

    int x = randIntBetween(min, max);
    int y = x;

    while (y == x) {
        y = randIntBetween(min, max);
    }

    if (x < y) {
        n1 = x;
        n2 = y;
    } else {
        n1 = y;
        n2 = x;
    }

}

int Random::discreteDistribution(int min, int max) {
    return binomialDistribution(max, 0.5) + min;
}

long int Random::generateSeed() {

    // http://cboard.cprogramming.com/cplusplus-programming/63995-cast-time_t-int.html
    time_t the_time; // declare a time_t variable
    long int x; // declare a long int
    the_time = time(0); // initialize the time_t variable with the system time
    x = static_cast<int>(the_time); // case the time_t variable to a long int

    return x;

}

/*
 * Private Methods
 */

int Random::uniformIntDiscreteDistribution(int min, int max) {

    // Uniform discrete distribution
    // http://www.cplusplus.com/reference/random/uniform_int_distribution/

    // http://www.cplusplus.com/reference/random/uniform_int_distribution/operator()/
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_int_distribution<int> distribution(min, max);

    return distribution(generator);

}

int Random::binomialDistribution(int max, float p) {

    //http://www.cplusplus.com/reference/random/binomial_distribution/operator()/

    // construct a trivial random generator engine from a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::binomial_distribution<int> distribution(max, p);
    return distribution(generator);
    
}

