#include "util.h"

int isMultiple(int x, int y) {
    int total = (int) x/y;
    return x==total*y;
}
readMatReturn readMat(FILE *f) {
  readMatReturn r;
  r.isError = -1;
  long long words, size;
  float* M;
  char* vocab;
  long long a, b;
  float len;
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  printf("reading file with words: %lld and size: %lld\n", words, size);
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return r;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;

    if( isMultiple(b, words/10) ) {
        printf("read %lld words\n", b);
    }

  }
  printf("read %lld words\n", b);
  r.words = words;
  r.size = size;
  r.M = M;
  r.vocab = vocab;
  r.isError = 0;
  return r;
}

