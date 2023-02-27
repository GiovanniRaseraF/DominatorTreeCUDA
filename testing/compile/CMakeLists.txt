cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(
    cmake_and_cuda
    VERSION 1.0 
    LANGUAGES CXX CUDA
)
# OpenMP
add_compile_options(-Wall -Wextra -Xcompiler -fopenmp)

add_executable(
    compile_test 
    compile.cu
)


target_compile_features(compile_test PUBLIC cxx_std_20)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    target_link_libraries(compile_test PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(
    TARGET compile_test 
    PROPERTY 
    CUDA_SEPARABLE_COMPILATION ON
    #CUDA_ARCHITECTURES OFF
)
