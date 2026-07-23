cmake_minimum_required(VERSION 3.15)
project(MeshDIC VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /bigobj /MP)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Eigen
set(EIGEN3_INCLUDE_DIR "C:/Users/lbd/miniconda3/Library/include/eigen3" CACHE PATH
    "Path to Eigen headers")
if(EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    set(eigen_SOURCE_DIR "${EIGEN3_INCLUDE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(Eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(Eigen)
    endif()
endif()

# OpenCV
set(OpenCV_DIR "C:/Users/lbd/miniconda3/Library")
set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
file(GLOB OpenCV_CORE_LIB "${OpenCV_DIR}/lib/opencv_core*.lib")
file(GLOB OpenCV_IMGPROC_LIB "${OpenCV_DIR}/lib/opencv_imgproc*.lib")

if(NOT OpenCV_CORE_LIB)
    message(FATAL_ERROR "OpenCV core library not found in ${OpenCV_DIR}/lib")
endif()

pybind11_add_module(_core
    src/cpp/bindings.cpp
    src/cpp/mesh_dic.cpp
    src/cpp/bspline.cpp
    src/cpp/local_icgn.cpp
    src/cpp/shape_func.cpp
    src/cpp/global2local.cpp
    src/cpp/stiffness.cpp
    src/cpp/strain.cpp
)

target_include_directories(_core PRIVATE
    ${eigen_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

target_link_libraries(_core PRIVATE
    ${OpenCV_CORE_LIB}
    ${OpenCV_IMGPROC_LIB}
)

target_compile_definitions(_core PRIVATE MESHDIC_HAS_OPENCV)

install(TARGETS _core DESTINATION mesh_dic)
