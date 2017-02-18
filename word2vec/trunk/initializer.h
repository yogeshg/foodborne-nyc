#ifndef __INITIALIZER_H
#define __INITIALIZER_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "util.h"

class BinReader {

public:
    BinReader(std::string fname):words(0), size(0), vocab(), model(), filename(fname), isRead(false), logger(std::cout) {};
    void read();                                // populates words, size, vocab and model
    long long words;                       // get number of words available after init
    long long size;                        // get size of each vector, available after init
    std::vector<std::string> vocab;        // get voabulary, populated after read()
    std::vector<float> model;              // get model, populated after read()
    ~BinReader();

private:
    std::string filename;
    bool isRead;
    std::ostream& logger;
    readMatReturn r;
 
};

class VocabInitializer {

public:
    VocabInitializer(std::string fname) : br(fname), logger(std::cout) {};
    void read() {logger<<"reading\n"; br.read(); makeWord2index();};
    std::vector<float> get(const std::string& word);
    long long getSize() { return br.size; };
    long long getWords() { return br.words; };

private:
    BinReader br;
    std::map<std::string, float> word2index;
    void makeWord2index();
    std::ostream& logger;
};








#endif /*__INITIALIZER_H */
