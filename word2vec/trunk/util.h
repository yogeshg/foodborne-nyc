#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#include "constants.h"

int isMultiple(int x, int y);
typedef struct _readMatReturn {
    long long words;
    long long size;
    char* vocab;
    float* M;
    int isError;
} readMatReturn;

readMatReturn readMat(FILE *f);
#endif /* UTIL_H */

