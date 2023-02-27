cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(
    thread
    VERSION 1.0 
    LANGUAGES CXX CUDA
)
# OpenMP
add_compile_options(-Wall -Wextra -Xcompiler -fopenmp)

add_executable(
    ${PROJECT_NAME}.out 
    ${PROJECT_NAME}.cu
)

target_compile_features(${PROJECT_NAME}.out PUBLIC cxx_std_20)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}.out PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET ${PROJECT_NAME}.out PROPERTY CUDA_ARCHITECTURES OFF)
set_property(
    TARGET ${PROJECT_NAME}.out 
    PROPERTY 
    CUDA_SEPARABLE_COMPILATION ON
)
