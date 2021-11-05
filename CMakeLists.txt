cmake_minimum_required(VERSION 3.15...3.19)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(dpu_kmeans VERSION ${VERSION})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# =================== BUILDING THE DPU BINARY ======================

include(ExternalProject)

ExternalProject_Add(
  dpu_program
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_program
  INSTALL_DIR ./dpu_program
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DNR_TASKLETS=${NR_TASKLETS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  BUILD_ALWAYS TRUE
  USES_TERMINAL_BUILD TRUE)

ExternalProject_Get_Property(dpu_program INSTALL_DIR)
set(DPU_PROGRAM_INSTALL_DIR ${INSTALL_DIR})

install(DIRECTORY ${DPU_PROGRAM_INSTALL_DIR} DESTINATION .)

# =================== BUILDING THE HOST BINARY ======================

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

set(PKG_CONFIG_EXECUTABLE "dpu-pkg-config")
find_package(PkgConfig REQUIRED)
pkg_search_module(DPU REQUIRED IMPORTED_TARGET dpu)

find_package(OpenMP REQUIRED)

pybind11_add_module(
  _core
  MODULE
  src/main.cpp
  src/trivial_checksum_example_host.c
  src/kmeans.c
  src/cluster.c
  src/kmeans_clustering.c
  src/kmeans_dpu.c
  src/rmse.c)
target_link_libraries(_core PRIVATE PkgConfig::DPU OpenMP::OpenMP_CXX
                                    OpenMP::OpenMP_C)
target_include_directories(_core PRIVATE PkgConfig::DPU)

target_compile_definitions(
  _core
  PUBLIC VERSION_INFO=${PROJECT_VERSION}
  PUBLIC NR_TASKLETS=${NR_TASKLETS})

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

set(CMAKE_C_FLAGS_RELEASE "-Ofast")

install(TARGETS _core DESTINATION .)
