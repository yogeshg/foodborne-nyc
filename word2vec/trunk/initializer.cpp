#include "initializer.h"
#include "stdio.h"

void VocabInitializer::makeWord2index() {
    for( long long b=0; b < br.words; ++b ) {
        word2index[br.vocab.at(b)] = b;
    }
}

std::vector<float> VocabInitializer::get(const std::string& word) {
    std::vector<float> vec;
    auto it = word2index.find(word);
    if( it!= word2index.end() ) {
        const long long b = it->second;
        // logger << "found word " << word << " at index " << b << "\n";
        for( long long a=0; a<br.size; ++a ) {
            vec.push_back( br.model.at(a + b * br.size));
        }
    }

    return vec;
}

void BinReader::read() {
    if( !isRead ) {
        FILE *f;
        f = fopen(filename.c_str(), "rb");
        if (f == NULL) {
            logger << "error opening file" << filename << "\n";
        } else {
            logger << "opened file " << filename << "\n";
            r = readMat(f);

            words = r.words;
            size = r.size;
            for(long long b=0; b<words; ++b) {
                vocab.push_back(&r.vocab[b*max_w]);
                // logger << "pushed " << vocab.at(b) <<"\n";
                for( long long a=0; a<size; ++a) {
                    model.push_back( r.M[a + b*size] );
                }
            }
            fclose( f );
            isRead = true;
        }
    }
}
BinReader::~BinReader() {
    if( isRead ) {
        free( r.vocab );
        free( r.M );
    }
}
