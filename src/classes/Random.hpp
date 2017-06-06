/* 
 * File:   Random.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on July 22, 2014, 9:23 PM
 */

#ifndef RANDOM_HPP
#define	RANDOM_HPP

#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <ctime>

#include "Util.hpp"

using namespace std;

class Random {

public:

    static void randomize();
    static double randDoubleBetween(double x0, double x1);
    static float randFloat();
    static int randInt();
    static int randIntBetween(int x0, int x1);
    static void randPermutation(int size, int min, vector<int>& elements);
    static void randTwoNumbers(int min, int max, int& n1, int& n2);
    static int discreteDistribution(int min, int max);
    static long int generateSeed();
    
private:

    Random();
    
    static int uniformIntDiscreteDistribution(int min, int max);
    static int binomialDistribution(int max, float p);

};

#endif	/* RANDOM_HPP */

