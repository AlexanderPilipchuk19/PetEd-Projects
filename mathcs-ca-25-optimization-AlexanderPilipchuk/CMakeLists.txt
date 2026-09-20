cmake_minimum_required(VERSION 3.10)
project(MonteCarloVolume CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(monte_carlo
    main.cpp
    hit.cpp
    hit.h
)

target_link_libraries(monte_carlo PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(monte_carlo PRIVATE
    $<$<CONFIG:Release>:-O3>
)

set_target_properties(monte_carlo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
