cmake_minimum_required(VERSION 3.10.2)
project(CPSpatial LANGUAGES CXX)

set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(OpenCL REQUIRED)

add_executable(cpspatial ${SRC_FILES})
target_link_libraries(cpspatial /usr/local/lib/librocm-dbgapi.so OpenCL::OpenCL)


