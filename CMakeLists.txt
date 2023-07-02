cmake_minimum_required(VERSION 3.21)

if (NOT DEFINED CMAKE_CXX_STANDARD)
 set(CMAKE_CXX_STANDARD 20)
endif()

project(edgefriend LANGUAGES CXX C)

add_subdirectory(rapidobj)
add_subdirectory(glm)

add_library(edgefriend)
target_include_directories(edgefriend PUBLIC include)
target_sources(edgefriend PRIVATE include/edgefriend.h src/edgefriend.cpp)
target_link_libraries(edgefriend PUBLIC glm::glm)

add_executable(edgefriend_demo)
target_sources(edgefriend_demo PRIVATE src/demo.cpp)
target_link_libraries(edgefriend_demo rapidobj edgefriend)
