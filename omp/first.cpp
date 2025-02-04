#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main(){
    // Utilizzo dei threads
    #pragma omp parallel num_threads(2)
    {
        std::cout << "Codice parallelo eseguito da " << omp_get_num_threads() << std::endl;
        auto tnum = omp_get_thread_num();
        if(tnum == 2){
            std::cout << "Ciao sono: " << tnum << " e faccio cose diverse" << std::endl;
        }else{
            std::cout << "Sono: " << tnum << std::endl;
        }
    }



    
    std::cout << std::endl;
    std::cout << "Reduction" << std::endl;
    // Reduction
    auto const NUM_T =  22;
    auto const LEN_SUBV =  10000;
    std::vector<int> myvec{};
    for(int i = 0; i < NUM_T; i++){
        for(int j = 0; j < LEN_SUBV; j++){
            myvec.push_back(i+j);
        }
    }

    int sum = 0;
    int tnum = 0;
    int start_pos = 0;
    // Start parallel
    #pragma omp parallel reduction(+ : sum) num_threads(NUM_T) shared(myvec) private(tnum, start_pos)
    {
        tnum = omp_get_thread_num();
        start_pos = tnum * LEN_SUBV;

        for(int i = 0; i < LEN_SUBV; i++){
            sum += myvec[i+start_pos];
        }

        // #pragma omp critical (stampa)
        // {std::cout << "Sono: " << tnum << " Sum: " << sum << std::endl << "start: " << start_pos << " end: "  <<
        // (start_pos + LEN_SUBV) << std::endl;}
    }

    std::cout << "Par Sum: " << sum << std::endl;

    auto sumnormal = std::reduce(myvec.cbegin(), myvec.cend());

    std::cout << "Seq Sum: " << sumnormal << std::endl;

    int sum2 = 0;
    // omp for
    // Start for
    #pragma omp parallel reduction(+ : sum2) num_threads(NUM_T) shared(myvec) private(tnum, start_pos)
    {
        sum2 = 0;
        #pragma omp for
        for(int i = 0; i < LEN_SUBV; i++){
            sum2 += myvec[i];
        }

        // #pragma omp critical (stampa)
        // {std::cout << "Sono: " << " Sum: " << sum2 << std::endl;}
    }

    std::cout << "Par Sum2: " << sum2 << std::endl;
}