cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_sequential_naive LANGUAGES CXX )

add_executable(
    sequential_naive 
    naive.cpp)

target_compile_features(sequential_naive PUBLIC cxx_std_23)