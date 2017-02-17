//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#include "constants.h"
#include "util.hpp"

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  long long words, size;
  float *M;
  char *vocab;
  readMatReturn r;
  if (argc < 2) {
    printf("Usage: ./word-analogy <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  r = readMat(f);
  if( r.isError ) {
    return -1;
  }
  words = r.words;
  size  = r.size ;
  M     = r.M    ;
  vocab = r.vocab;

  fclose(f);
  return 0;
}


