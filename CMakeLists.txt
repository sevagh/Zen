cmake_minimum_required(VERSION 3.14)
project(rhythm_toolkit)
set(VERSION 0.0.1)

include(CTest)
enable_testing()

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(ExternalProject)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
