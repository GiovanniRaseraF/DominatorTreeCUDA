cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(
    thread_test 
    thread.cu)

target_compile_features(thread_test PUBLIC cxx_std_20)
set_property(TARGET thread_test
             PROPERTY 
             CUDA_SEPARABLE_COMPILATION ON)