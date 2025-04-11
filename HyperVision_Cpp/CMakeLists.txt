cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(HyperVision)
set(CMAKE_CXX_STANDARD 14)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(flow_construct)
add_subdirectory(dataset_construct)
add_subdirectory(packet_parse)
add_subdirectory(graph_analyze)

# Create a shared library that contains all the components
# Create an empty source file for the library
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "")
add_library(hypervision_lib SHARED ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
target_link_libraries(hypervision_lib 
    graph_analysis
    flow_construct
    dataset_construct
    packet_parse
    gflags
    ${OpenMP_CXX_LIBRARIES}
)

# Main executable
aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} "${DIR_SRCS}")

target_link_libraries(${PROJECT_NAME} hypervision_lib)

# Add performance test directory
add_subdirectory(performance_test)
