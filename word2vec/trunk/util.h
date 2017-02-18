#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <string>
#include <sstream>
#include <string.h>
#include <math.h>
#include <malloc.h>

#include "constants.h"
#include <iostream>

int isMultiple(int x, int y);
typedef struct _readMatReturn {
    long long words;
    long long size;
    char* vocab;
    float* M;
    int isError;
} readMatReturn;

readMatReturn readMat(FILE *f);

template <class Container>
void toString(Container c, std::ostream& ss, int max=10) {
    for(auto x: c){
        ss<< x << " ";
        if(!--max) {
            ss << "...";
            break;
        }
    }
}
template <class Container>
std::string toString (Container c, int max=10) {
    std::stringstream ss;
    toString(c, ss, max);
    return ss.str();
}
template <class Container>
void print(Container c) {
    toString(c, std::cout);
}

#endif /* UTIL_H */

