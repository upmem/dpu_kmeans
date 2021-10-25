cmake_minimum_required(VERSION 3.15...3.19)

project(dpu_program)

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

add_executable(helloworld helloworld.c)
add_executable(trivial_checksum_example trivial_checksum_example.c)

install(TARGETS helloworld DESTINATION .)
install(TARGETS trivial_checksum_example DESTINATION .)
