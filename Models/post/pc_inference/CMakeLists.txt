set(CMAKE_C_COMPILER "${CMAKE_CROSS_COMPILER}gcc")
set(CMAKE_CXX_COMPILER "${CMAKE_CROSS_COMPILER}g++")

project(magikExecutor)
cmake_minimum_required(VERSION 3.1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -std=c++11 -lrt -pthread -mavx -ldl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g -std=c++11 -pthread -ldl")

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/../lib)

add_executable(pc_inference_bin inference.cc)
#target_link_libraries(pc_inference_bin -Wl,-whole-archive libmagikExecutor.a -Wl,-no-whole-archive libprotobuf.a libprotobuf-lite.a)
target_link_libraries(pc_inference_bin libmagikExecutor.so libprotobuf.so libprotobuf-lite.so)
