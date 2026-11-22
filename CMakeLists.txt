cmake_minimum_required(VERSION 3.20)
project(GeneticNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

add_executable(gen_nn src/main.cpp)

find_package(Threads REQUIRED)
target_link_libraries(gen_nn PRIVATE Threads::Threads)
