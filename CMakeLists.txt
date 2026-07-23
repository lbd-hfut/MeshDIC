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
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    if(DEFINED ENV{CONDA_PREFIX} AND WIN32)
        set(EIGEN3_INCLUDE_DIR "$ENV{CONDA_PREFIX}/Library/include/eigen3" CACHE PATH
            "Path to Eigen headers")
    elseif(DEFINED ENV{CONDA_PREFIX})
        set(EIGEN3_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/eigen3" CACHE PATH
            "Path to Eigen headers")
    else()
        set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path to Eigen headers")
    endif()
endif()

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
if(NOT DEFINED OpenCV_DIR)
    if(DEFINED ENV{CONDA_PREFIX} AND WIN32)
        set(OpenCV_DIR "$ENV{CONDA_PREFIX}/Library" CACHE PATH "Path to OpenCV")
    elseif(DEFINED ENV{CONDA_PREFIX})
        set(OpenCV_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Path to OpenCV")
    else()
        set(OpenCV_DIR "" CACHE PATH "Path to OpenCV")
    endif()
endif()

set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
if(WIN32)
    file(GLOB OpenCV_CORE_LIB "${OpenCV_DIR}/lib/opencv_core*.lib")
    file(GLOB OpenCV_IMGPROC_LIB "${OpenCV_DIR}/lib/opencv_imgproc*.lib")
else()
    file(GLOB OpenCV_CORE_LIB "${OpenCV_DIR}/lib/libopencv_core*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    file(GLOB OpenCV_IMGPROC_LIB "${OpenCV_DIR}/lib/libopencv_imgproc*${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

if(NOT OpenCV_CORE_LIB)
    message(FATAL_ERROR "OpenCV core library not found in ${OpenCV_DIR}/lib")
endif()
if(NOT OpenCV_IMGPROC_LIB)
    message(FATAL_ERROR "OpenCV imgproc library not found in ${OpenCV_DIR}/lib")
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
