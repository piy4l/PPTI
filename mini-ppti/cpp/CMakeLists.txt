cmake_minimum_required(VERSION 3.16)
project(mini_ppti CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE CONFIG REQUIRED PATHS /usr/local)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(mini_ppti
    src/main.cpp
    src/ckks_runner.cpp
    src/profiler.cpp
)

target_include_directories(mini_ppti PRIVATE
    include
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

target_link_directories(mini_ppti PRIVATE
    /usr/local/lib
    /opt/homebrew/opt/libomp/lib
)

target_link_libraries(mini_ppti PRIVATE
    OPENFHEpke
    OPENFHEcore
    OPENFHEbinfhe
    OpenMP::OpenMP_CXX
)