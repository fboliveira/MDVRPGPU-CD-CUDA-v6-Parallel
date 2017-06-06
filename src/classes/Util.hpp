/*
* File:   Util.hpp
* Author: Fernando B Oliveira <fboliveira25@gmail.com>
*
* Created on July 23, 2014, 12:21 AM
*/

#ifndef UTIL_H
#define	UTIL_H

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <iostream>

#include "../global.hpp"

using namespace std;

class Util {

public:
    Util();

    template<class Iter>
    static int findValueInVector(Iter begin, Iter end, float value);

    template<typename Iter>
    static int countElementsInVector(Iter begin, Iter end, int value);

    static bool isBetterSolution(float newCost, float currentCost);
    static bool isBetterOrEqualSolution(float newCost, float currentCost);
    static bool isEqualSolution(float newCost, float currentCost);

    static float scaledFloat(float value);
    static float calculateGAP(float cost, float bestKnowSolution);
    static double diffTimeFromStart(time_t start);

    static void printTimeNow();
    static void error(const string message, int num);

    static void print(vector<int>& vector);
    static void print(int *vector, int size);
    static void print(int *vector, int first, int last);

    template<typename Iter>
    static void print(Iter begin, Iter end);

    static void change(int* va, int x, int *vb, int y);
    static void insert(int* array, int size, int position, int value);
    static void remove(int* array, int size, int position);

    static void selectVectorOrder(vector<typedef_order>& vector);
    static float calculateEucDist2D(float x1, float y1, float x2, float y2);

private:


};

#endif	/* UTIL_H */

