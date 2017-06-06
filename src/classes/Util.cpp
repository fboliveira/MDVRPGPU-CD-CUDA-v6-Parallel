/*
* File:   Util.cpp
* Author: Fernando B Oliveira <fboliveira25@gmail.com>
*
* Created on July 23, 2014, 12:21 AM
*/

#include "Util.hpp"
#include "AlgorithmConfig.hpp"
#include "Route.hpp"
#include "LocalSearch.hpp"

/*
* Constructors
*/

Util::Util() {
}

/*
* Getters and Setters
*/



/*
* Methods
*/

template<class Iter>
int Util::findValueInVector(Iter begin, Iter end, float value) {

    // 
    // http://stackoverflow.com/questions/18138075/vector-iterator-parameter-as-template-c

    // http://stackoverflow.com/questions/571394/how-to-find-an-item-in-a-stdvector
    // find item
    // std::find(vector.begin(), vector.end(), item)!=vector.end()    

    // http://stackoverflow.com/questions/15099707/how-to-get-position-of-a-certain-element-in-strings-vector-to-use-it-as-an-inde

    Iter it = find(begin, end, value);

    if (it == end)
        return -1;
    else
        return it - begin;

}

template<typename Iter>
int Util::countElementsInVector(Iter begin, Iter end, int value) {
    // http://www.cplusplus.com/reference/algorithm/count/
    return std::count(begin, end, value);
}

bool Util::isBetterSolution(float newCost, float currentCost) {

    if (scaledFloat(newCost) >= 0 && scaledFloat(newCost) < scaledFloat(currentCost)) // || ((newCost - currentCost) < 0.001))
        return true;
    else
        return false;

}


bool Util::isBetterOrEqualSolution(float newCost, float currentCost) {

    if (scaledFloat(newCost) >= 0 && scaledFloat(newCost) <= scaledFloat(currentCost)) // || ((newCost - currentCost) < 0.001))
        return true;
    else
        return false;

}

bool Util::isEqualSolution(float newCost, float currentCost) {
    if (scaledFloat(newCost) >= 0 && scaledFloat(newCost) == scaledFloat(currentCost)) // || ((newCost - currentCost) < 0.001))
        return true;
    else
        return false;
}

float Util::scaledFloat(float value) {

    if (value == FLT_MAX)
        return value;

    long long int scaled = value * 100;
    value = static_cast<float>(scaled) / 100.0;
    return value;

}

float Util::calculateGAP(float cost, float bestKnowSolution) {
    return ((cost - bestKnowSolution) / bestKnowSolution) * 100.00;
}

double Util::diffTimeFromStart(time_t start) {

    time_t end;
    // double difftime (time_t end, time_t beginning);
    // Return difference between two times
    time(&end);

    return difftime(end, start);
}

void Util::printTimeNow() {

    const time_t ctt = time(0);
    cout << asctime(localtime(&ctt)) << "\n";

}

void Util::error(const string message, int num) {

    cout << "\n\nERRO: ==========================================\n";
    cout << "MSG: " << message << " - " << num << endl;
    cout << "================================================\n\n";

}

void Util::print(vector<int>& vector) {
    print(vector.begin(), vector.end());
}

template<typename Iter>
void Util::print(Iter begin, Iter end) {

    size_t size = end - begin;

    cout << "Size: " << size << " => ";

    for (Iter iter = begin; iter != end; ++iter) {
        cout << *iter;

        if (next(iter) != end)
            cout << " - ";
    }

    cout << endl;
}

void Util::print(int *vector, int size) {
    Util::print(vector, 0, size - 1);
}

void Util::print(int *vector, int first, int last) {

    cout << "Size: " << last - first + 1 << " => ";

    for (int i = first; i <= last; ++i) {
        cout << vector[i];

        if (i + 1 <= last)
            cout << " - ";
    }

    cout << endl;

}


// Change va[x] with va[y]
void Util::change(int* va, int x, int *vb, int y) {

    int aux = vb[y];
    vb[y] = va[x];
    va[x] = aux;

}

void Util::insert(int* array, int size, int position, int value) {

    for (int i = size - 1; i >= position; --i)
        array[i + 1] = array[i];

    array[position] = value;

}

void Util::remove(int* array, int size, int position) {

    for (int i = position; i < size - 1; ++i)
        array[i] = array[i + 1];

}

void Util::selectVectorOrder(vector<typedef_order>& vector) {

    size_t min, i, j;
    typedef_order aux;

    for (i = 0; i < vector.size() - 1; ++i) {

        min = i;
        // Procura pelo menor elemento

        for (j = i + 1; j < vector.size(); ++j) {

            if (vector.at(j).cost < vector.at(min).cost)
                min = j;
        }

        // Substitui o menor elemento com o elemento em i
        aux = vector.at(min);
        vector.at(min) = vector.at(i);
        vector.at(i) = aux;
    }
}

float Util::calculateEucDist2D(float x1, float y1, float x2, float y2) {

    float xd = x1 - x2;
    float yd = y1 - y2;

    return sqrtf(xd * xd + yd * yd);
}

//float Util::calculateEucDist2D(int x1, int y1, int x2, int y2) {
//
//    int xd = x1 - x2;
//    int yd = y1 - y2;
//
//    return sqrtf(xd * xd + yd * yd);
//}
