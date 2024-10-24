cmake_minimum_required(VERSION 3.15...3.27)

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

include(CheckIPOSupported)

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be set")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g0")

add_executable(trivial_checksum_example trivial_checksum_example.c)
add_executable(kmeans_dpu_kernel kmeans_dpu_kernel.c)

target_compile_definitions(kmeans_dpu_kernel PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(kmeans_dpu_kernel PUBLIC -DNR_TASKLETS=${NR_TASKLETS})

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

install(TARGETS trivial_checksum_example DESTINATION ${PROJECT_NAME})
install(TARGETS kmeans_dpu_kernel DESTINATION ${PROJECT_NAME})

# copying the compilation database for language servers
add_custom_target(
  copy-compile-commands ALL
  ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_CURRENT_LIST_DIR})
