cmake_minimum_required(VERSION 3.8)
project(RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu")

file(GLOB_RECURSE EXTERNAL_FILES
    "${CMAKE_SOURCE_DIR}/src/external/*.cpp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(RayTracing ${SRC_FILES} ${EXTERNAL_FILES})

set_target_properties(RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
