cmake_minimum_required(VERSION 3.15...3.27)

project(dpu_program LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

include(CheckIPOSupported)
include(../../cmake/CompilerWarnings.cmake)

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be set")
endif()

set(CHIP_VERSION "v1A") # Default value
if(EXISTS "/sys/class/dpu_rank/dpu_rank0/dpu_chip_id")
  file(READ "/sys/class/dpu_rank/dpu_rank0/dpu_chip_id" CHIP_ID_NUMBER)
  if(CHIP_ID_NUMBER GREATER 8)
    set(CHIP_VERSION "v1B")
  endif()
endif()

add_executable(kmeans_dpu_kernel kmeans_dpu_kernel.c)
target_sources(
  kmeans_dpu_kernel
  PRIVATE FILE_SET
          common_headers
          TYPE
          HEADERS
          BASE_DIRS
          ../
          FILES
          ../common.h)

target_compile_options(kmeans_dpu_kernel PRIVATE -mcpu=${CHIP_VERSION}
                                                 $<$<CONFIG:DEBUG>:-Og>)
target_compile_definitions(kmeans_dpu_kernel PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(kmeans_dpu_kernel PUBLIC -mcpu=${CHIP_VERSION}
                    -DNR_TASKLETS=${NR_TASKLETS})
dpu_kmeans_set_project_warnings(kmeans_dpu_kernel "" "")

check_ipo_supported(
  RESULT ipo_supported
  OUTPUT error
  LANGUAGES C)
if(ipo_supported)
  message(STATUS "IPO / LTO enabled")
  set_target_properties(kmeans_dpu_kernel
                        PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
  message(FATAL_ERROR "IPO / LTO not supported: <${error}>")
endif()

install(TARGETS kmeans_dpu_kernel DESTINATION ${PROJECT_NAME})

# copying the compilation database for language servers
add_custom_target(
  copy-compile-commands ALL
  ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_CURRENT_LIST_DIR})
