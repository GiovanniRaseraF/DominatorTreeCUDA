#include <iostream>
#include <experimental/simd>
namespace stdx = std::experimental;
using intv  = stdx::native_simd<int>;

int main(){
    intv a([](int i){return i;});
    intv b([](int i){return i*3;});
    intv c = a+b;

    std::cout << "c[3]: " << c[3] << std::endl;
    return 0;
}