#include <iostream>
#include "initializer.h"
#include "vector"
#include "string"

int main() {
    BinReader br("vectors_sample.bin");
    br.read();
    std::cout << "br.words:       " << br.words << "\n";
    std::cout << "br.size:        " << br.size << "\n";
    std::cout << "br.vocab.size():" << br.vocab.size() << "\n";
    std::cout << "br.model.size():" << br.model.size() << "\n";

    VocabInitializer vi("vectors_sample.bin");
    vi.read();
    std::vector<std::string> words {"idea", "january", "king", "cities", "king", "yogesh"};

    for( auto w : words ) {
        std::cout << w << ": ";
        int printonly = 10;
        for( auto c : vi.get( w ) ) {
            std::cout << c << ", ";
            if(!--printonly) break;
        }
        std::cout << vi.get( w ).size() << "\n";

    }

}
