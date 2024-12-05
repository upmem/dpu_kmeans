cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be set")
endif()

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

# =================== BUILDING THE DPU BINARY ======================

include(ExternalProject)

ExternalProject_Add(
  dpu_program
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_program
  INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_kmeans
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DNR_TASKLETS=${NR_TASKLETS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    # temporary workaround until SDK distributes fixed toolchain file
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_program/dpu.cmake
  BUILD_ALWAYS TRUE)

# =================== BUILDING THE HOST BINARY ======================

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

include(CheckIPOSupported)
include(${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake)
include(cmake/CPM.cmake)
include(cmake/CompilerWarnings.cmake)

cpmaddpackage(
  NAME
  fmt
  GITHUB_REPOSITORY
  fmtlib/fmt
  GIT_TAG
  11.0.2
  OPTIONS
  "CMAKE_POSITION_INDEPENDENT_CODE True")

pybind11_add_module(_core MODULE src/main.cpp src/host_program/dimm_manager.cpp
                    src/host_program/lloyd_iter.cpp)
target_sources(
  _core
  PRIVATE FILE_SET
          common_headers
          TYPE
          HEADERS
          BASE_DIRS
          src
          FILES
          src/common.h)
target_sources(
  _core
  PRIVATE FILE_SET
          host_headers
          TYPE
          HEADERS
          BASE_DIRS
          src
          FILES
          src/kmeans.hpp)
target_link_libraries(_core PRIVATE ${DPU_HOST_LIBRARIES} fmt stdc++fs)
target_include_directories(_core SYSTEM PUBLIC ${DPU_HOST_INCLUDE_DIRECTORIES})
target_link_directories(_core PUBLIC ${DPU_HOST_LINK_DIRECTORIES})

target_compile_features(_core PUBLIC cxx_std_17)
target_compile_options(_core PRIVATE $<$<CONFIG:RELEASE>:-Ofast>
                                     $<$<CONFIG:DEBUG>:-Og>)
dpu_kmeans_set_project_warnings(_core "" "")

target_compile_definitions(
  _core
  PUBLIC VERSION_INFO=${SKBUILD_PROJECT_VERSION}
  PUBLIC NR_TASKLETS=${NR_TASKLETS})

check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
  message(STATUS "IPO / LTO enabled")
  set_target_properties(_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE
                                         TRUE)
else()
  message(STATUS "IPO / LTO not supported: <${error}>")
endif()

# install in-source rather than in the temporary build directory that way
# linting tools can find the compiled extension
install(TARGETS _core
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_kmeans)

# copying the compilation database for language servers
add_custom_target(
  copy-compile-commands ALL
  ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_CURRENT_LIST_DIR}/src)

# writing configuration file for clangd
list(GET pybind11_INCLUDE_DIRS 0 PYBIND11_INCLUDE)
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"
  OUTPUT_VARIABLE PYTHON_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(GET CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES 0 GCC_INCLUDE)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.clangd.in
               ${CMAKE_CURRENT_SOURCE_DIR}/.clangd @ONLY)
