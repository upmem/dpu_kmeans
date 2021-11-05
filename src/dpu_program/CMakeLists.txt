cmake_minimum_required(VERSION 3.15...3.19)

project(dpu_program)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG "-g -Og")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3 -flto=thin")

add_executable(trivial_checksum_example trivial_checksum_example.c)
add_executable(kmeans_dpu_kernel kmeans_dpu_kernel.c)

target_compile_definitions(kmeans_dpu_kernel PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(kmeans_dpu_kernel PUBLIC -DNR_TASKLETS=${NR_TASKLETS})

install(TARGETS trivial_checksum_example DESTINATION .)
install(TARGETS kmeans_dpu_kernel DESTINATION .)
