cmake_minimum_required(VERSION 3.15...3.19)

project(dpu_trees)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =================== BUILDING THE DPU BINARY ======================

include(ExternalProject)

ExternalProject_Add(
  dpu_program
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_program
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>/dpu_program)

ExternalProject_Get_Property(dpu_program INSTALL_DIR)
set(DPU_PROGRAM_INSTALL_DIR ${INSTALL_DIR}/dpu_program)

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
# if(DEFINED ENV{UPMEM_HOME}) list(APPEND CMAKE_PREFIX_PATH
# "${UPMEM_HOME}/share/pkgconfig") # set(ENV{PKG_CONFIG_PATH}
# "${UPMEM_HOME}/share/pkgconfig") endif()
pkg_search_module(DPU REQUIRED IMPORTED_TARGET dpu)

pybind11_add_module(_core MODULE src/main.cpp src/trees.c)
target_link_libraries(_core PRIVATE PkgConfig::DPU)
target_include_directories(_core PRIVATE PkgConfig::DPU)

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION .)
