cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(
    compile_test 
    compile.cu)

target_compile_features(compile_test PUBLIC cxx_std_20)
set_property(TARGET compile_test 
             PROPERTY 
             CUDA_SEPARABLE_COMPILATION ON)