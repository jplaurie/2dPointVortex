cmake_minimum_required(VERSION 3.20)
project(PointVortex VERSION 0.1.0 LANGUAGES CXX)
option(POINT_VORTEX_OPENMP "Enable OpenMP acceleration" ON)
option(POINT_VORTEX_MPI "Build the MPI backend" ON)
option(POINT_VORTEX_CUDA "Build the CUDA backend" ON)
option(POINT_VORTEX_CUDA_TESTS "Run GPU-backed CUDA tests with CTest" OFF)
set(POINT_VORTEX_CUDA_ARCHITECTURES "120" CACHE STRING
    "CUDA architectures to build; defaults to NVIDIA compute capability 12.0")
set(POINT_VORTEX_LIBRARY_SOURCES
    src/compute.cpp src/timestep.cpp src/read.cpp src/checkpoint.cpp src/dipole.cpp
    src/backend_common.cpp src/simulation.cpp initial_conditions/initial_condition.cpp)
add_library(point_vortex_lib ${POINT_VORTEX_LIBRARY_SOURCES})
target_include_directories(point_vortex_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/initial_conditions)
target_compile_features(point_vortex_lib PUBLIC cxx_std_20)
add_library(point_vortex_lib_serial ${POINT_VORTEX_LIBRARY_SOURCES})
target_include_directories(point_vortex_lib_serial PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                          ${CMAKE_CURRENT_SOURCE_DIR}/initial_conditions)
target_compile_features(point_vortex_lib_serial PUBLIC cxx_std_20)
if(POINT_VORTEX_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(point_vortex_lib PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(point_vortex_lib PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(point_vortex_lib_serial PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_executable(point_vortex_cpu src/main.cpp src/print.cpp src/backend_cpu.cpp)
target_link_libraries(point_vortex_cpu PRIVATE point_vortex_lib)
# Preserve the original executable name as an alias-like second CPU target.
add_executable(point_vortex src/main.cpp src/print.cpp src/backend_cpu.cpp)
target_link_libraries(point_vortex PRIVATE point_vortex_lib)

if(POINT_VORTEX_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    add_executable(point_vortex_mpi src/main.cpp src/print.cpp src/backend_mpi.cpp)
    target_link_libraries(point_vortex_mpi PRIVATE point_vortex_lib MPI::MPI_CXX)
  else()
    message(WARNING "MPI was not found; point_vortex_mpi will not be built")
  endif()
endif()

if(POINT_VORTEX_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(point_vortex_cuda src/main.cpp src/print.cpp src/backend_cuda.cu)
    target_link_libraries(point_vortex_cuda PRIVATE point_vortex_lib)
    if(NOT POINT_VORTEX_CUDA_ARCHITECTURES STREQUAL "")
      set_property(TARGET point_vortex_cuda PROPERTY
                   CUDA_ARCHITECTURES "${POINT_VORTEX_CUDA_ARCHITECTURES}")
    endif()
    get_target_property(_pv_cuda_architectures point_vortex_cuda CUDA_ARCHITECTURES)
    message(STATUS "PointVortex CUDA architectures: ${_pv_cuda_architectures}")
    set_target_properties(point_vortex_cuda PROPERTIES
                                                       CUDA_STANDARD 20
                                                       CUDA_STANDARD_REQUIRED ON)
    add_executable(point_vortex_cuda_mixed src/main.cpp src/print.cpp src/backend_cuda_mixed.cu)
    target_link_libraries(point_vortex_cuda_mixed PRIVATE point_vortex_lib)
    if(NOT POINT_VORTEX_CUDA_ARCHITECTURES STREQUAL "")
      set_property(TARGET point_vortex_cuda_mixed PROPERTY
                   CUDA_ARCHITECTURES "${POINT_VORTEX_CUDA_ARCHITECTURES}")
    endif()
    set_target_properties(point_vortex_cuda_mixed PROPERTIES
                          CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  else()
    message(WARNING "CUDA was not found; CUDA backends will not be built")
  endif()
endif()
add_executable(point_vortex_benchmark src/benchmark.cpp)
target_link_libraries(point_vortex_benchmark PRIVATE point_vortex_lib)
add_executable(point_vortex_backend_benchmark_cpu src/backend_benchmark.cpp src/backend_cpu.cpp)
target_link_libraries(point_vortex_backend_benchmark_cpu PRIVATE point_vortex_lib)
add_executable(point_vortex_backend_benchmark_serial src/backend_benchmark.cpp src/backend_cpu.cpp)
target_link_libraries(point_vortex_backend_benchmark_serial PRIVATE point_vortex_lib_serial)
if(TARGET point_vortex_mpi)
  add_executable(point_vortex_backend_benchmark_mpi src/backend_benchmark.cpp src/backend_mpi.cpp)
  target_link_libraries(point_vortex_backend_benchmark_mpi PRIVATE point_vortex_lib MPI::MPI_CXX)
endif()
if(TARGET point_vortex_cuda)
  add_executable(point_vortex_backend_benchmark_cuda src/backend_benchmark.cpp src/backend_cuda.cu)
  target_link_libraries(point_vortex_backend_benchmark_cuda PRIVATE point_vortex_lib)
  if(NOT POINT_VORTEX_CUDA_ARCHITECTURES STREQUAL "")
    set_property(TARGET point_vortex_backend_benchmark_cuda PROPERTY
                 CUDA_ARCHITECTURES "${POINT_VORTEX_CUDA_ARCHITECTURES}")
  endif()
  set_target_properties(point_vortex_backend_benchmark_cuda PROPERTIES
                        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()
if(TARGET point_vortex_cuda_mixed)
  add_executable(point_vortex_backend_benchmark_cuda_mixed
                 src/backend_benchmark.cpp src/backend_cuda_mixed.cu)
  target_link_libraries(point_vortex_backend_benchmark_cuda_mixed PRIVATE point_vortex_lib)
  if(NOT POINT_VORTEX_CUDA_ARCHITECTURES STREQUAL "")
    set_property(TARGET point_vortex_backend_benchmark_cuda_mixed PROPERTY
                 CUDA_ARCHITECTURES "${POINT_VORTEX_CUDA_ARCHITECTURES}")
  endif()
  set_target_properties(point_vortex_backend_benchmark_cuda_mixed PROPERTIES
                        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()
add_executable(point_vortex_initial initial_conditions/generate_initial.cpp)
target_link_libraries(point_vortex_initial PRIVATE point_vortex_lib)
include(CTest)
if(BUILD_TESTING)
  add_executable(point_vortex_tests tests/tests.cpp)
  target_link_libraries(point_vortex_tests PRIVATE point_vortex_lib)
  add_test(NAME point_vortex_tests COMMAND point_vortex_tests)
  add_executable(point_vortex_audit_tests tests/audit_tests.cpp)
  target_link_libraries(point_vortex_audit_tests PRIVATE point_vortex_lib)
  add_test(NAME point_vortex_audit_tests COMMAND point_vortex_audit_tests)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME point_vortex_output_tests
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/output_integration.py
                     $<TARGET_FILE:point_vortex_cpu>)
    if(POINT_VORTEX_CUDA_TESTS)
      if(TARGET point_vortex_cuda)
        add_test(NAME point_vortex_cuda_tests
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/backend_consistency.py
                         $<TARGET_FILE:point_vortex_cpu>
                         $<TARGET_FILE:point_vortex_cuda>)
        set_tests_properties(point_vortex_cuda_tests PROPERTIES LABELS "cuda")
      endif()
      if(TARGET point_vortex_cuda_mixed)
        add_test(NAME point_vortex_cuda_mixed_tests
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/mixed_precision_consistency.py
                         $<TARGET_FILE:point_vortex_cpu>
                         $<TARGET_FILE:point_vortex_cuda_mixed>)
        set_tests_properties(point_vortex_cuda_mixed_tests PROPERTIES LABELS "cuda")
      endif()
    endif()
  endif()
endif()
