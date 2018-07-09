cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_FIND_ROOT_PATH "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_MODULE_PATH "${CMAKE_INSTALL_PREFIX}/share/cmake/modules")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel")
set(CMAKE_CXX_FLAGS "-Wall")

add_subdirectory(pysrc)
add_subdirectory(config)