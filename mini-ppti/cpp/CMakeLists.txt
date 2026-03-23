cmake_minimum_required(VERSION 3.16)
project(mini_ppti_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(mini_ppti
    src/main.cpp
    src/ckks_runner.cpp
    src/profiler.cpp
)