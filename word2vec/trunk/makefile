CC = gcc
CXX = g++ -std=c++11
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CXXFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

word2vec : word2vec.cpp initializer.o util.o

all: word2vec word2phrase distance word-analogy compute-accuracy

word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
distance2 : distance2.c
	$(CC) distance2.c -o distance2 $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

read-bin : read-bin.c
	$(CC) read-bin.c -o read-bin $(CFLAGS)

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy read-bin *.o a.out
