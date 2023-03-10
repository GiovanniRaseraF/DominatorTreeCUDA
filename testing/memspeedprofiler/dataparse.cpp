#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>

int main(){
    //                      bytes, htd, dth
    std::vector<std::tuple< int, float, float>> speeds;
    int bytes = 0;
    std::string line;
    bool iter = true; 

    while(iter){
        std::getline(std::cin, line);
        if(line == "" || line == "\n"){

        }else if(line == "endfile"){
            iter = false;
        }else{
            std::stringstream ss(line);
            ss >> bytes;
            std::tuple<int, float, float> newread;

            std::string title, scale, htd_v, dth_v;

            std::getline(std::cin, title);
            std::getline(std::cin, scale);
            std::getline(std::cin, htd_v);
            std::getline(std::cin, dth_v);

            std::stringstream htd_ss(htd_v);
            std::stringstream dth_ss(dth_v);
            float  htd, dth;
            htd_ss >> htd >> htd >> htd >> htd;
            dth_ss >> dth>> dth>> dth>> dth;

            std::get<0>(newread) = bytes;
            std::get<1>(newread) = htd;
            std::get<2>(newread) = dth;

            speeds.push_back(newread);
        }
    }
    
    std::cout << std::setw(15) << std::left << "bytes" << "," << std::setw(15) << std::left << "htd" << "," << std::setw(15) << std::right << "dth" << std::endl;
    for(auto newread : speeds){
        std::cout << std::setw(15) << std::left << std::get<0>(newread) << "," << std::setw(15) << std::left << std::get<1>(newread) << "," << std::setw(15) << std::right << std::get<2>(newread) << std::endl;
    }
}