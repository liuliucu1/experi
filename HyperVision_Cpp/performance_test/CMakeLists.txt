cmake_minimum_required(VERSION 3.10)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add the performance test executables
add_executable(performance_test performance_test_main.cpp)
add_executable(accuracy_test accuracy_test_main.cpp)

# Link with the main project libraries
# Note: hypervision_lib is created in the parent CMakeLists.txt
target_link_libraries(performance_test 
    hypervision_lib
    ${OpenMP_CXX_LIBRARIES}
)

target_link_libraries(accuracy_test 
    hypervision_lib
    ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(performance_test PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/graph_analyze
    ${CMAKE_SOURCE_DIR}/flow_construct
    ${CMAKE_SOURCE_DIR}/packet_parse
)

target_include_directories(accuracy_test PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/graph_analyze
    ${CMAKE_SOURCE_DIR}/flow_construct
    ${CMAKE_SOURCE_DIR}/packet_parse
)
