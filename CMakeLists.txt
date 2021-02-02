cmake_minimum_required(VERSION 3.18)

project(KMeans_Image_Clustering VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(KMeans_Image_Clustering main.cpp)

target_link_libraries(KMeans_Image_Clustering ${OpenCV_LIBS})
