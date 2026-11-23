cmake_minimum_required(VERSION 3.20)
project(GeneticNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(gen_nn src/main.cpp)

add_executable(gen_nn_test src/test.cpp)

target_include_directories(gen_nn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
)

target_include_directories(gen_nn_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
)

find_package(Threads REQUIRED)
target_link_libraries(gen_nn PRIVATE Threads::Threads)
