CXX ?= g++
CPPFLAGS ?= -Isrc -Iinitial_conditions
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic
OPENMP ?= -fopenmp
MPICXX ?= mpicxx
NVCC ?= nvcc
PYTHON ?= python3
# Leave empty to use nvcc's default; override for a specific target GPU.
CUDA_ARCH ?=
CUDA_ARCH_FLAGS = $(if $(strip $(CUDA_ARCH)),-arch=$(CUDA_ARCH),)
BUILD_DIR ?= build/make
OBJ_DIR := $(BUILD_DIR)/obj

LIB_SOURCES = src/compute.cpp src/timestep.cpp src/read.cpp src/checkpoint.cpp src/dipole.cpp \
	src/backend_common.cpp initial_conditions/initial_condition.cpp
SIMULATION_SOURCE = src/simulation.cpp
APP_SOURCES = src/main.cpp src/print.cpp
HEADERS = $(wildcard src/*.h initial_conditions/*.h)
LIB_OBJECTS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(LIB_SOURCES))
SIMULATION_OBJECT = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SIMULATION_SOURCE))
APP_OBJECTS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(APP_SOURCES))
CPU_OBJECT = $(OBJ_DIR)/src/backend_cpu.o
CUDA_OBJECT = $(OBJ_DIR)/src/backend_cuda.o
MIXED_CUDA_OBJECT = $(OBJ_DIR)/src/backend_cuda_mixed.o

.PHONY: all cpu mpi cuda cuda-mixed initial test test-output test-cuda-mixed benchmark \
	benchmark-backends clean
all: cpu initial
cpu: $(BUILD_DIR)/point_vortex $(BUILD_DIR)/point_vortex_cpu
mpi: $(BUILD_DIR)/point_vortex_mpi
cuda: $(BUILD_DIR)/point_vortex_cuda
cuda-mixed: $(BUILD_DIR)/point_vortex_cuda_mixed
initial: $(BUILD_DIR)/point_vortex_initial
benchmark: $(BUILD_DIR)/point_vortex_benchmark
benchmark-backends: $(BUILD_DIR)/point_vortex_backend_benchmark_cpu \
	$(BUILD_DIR)/point_vortex_backend_benchmark_serial \
	$(BUILD_DIR)/point_vortex_backend_benchmark_mpi \
	$(BUILD_DIR)/point_vortex_backend_benchmark_cuda \
	$(BUILD_DIR)/point_vortex_backend_benchmark_cuda_mixed

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

$(CUDA_OBJECT): src/backend_cuda.cu $(HEADERS) | $(OBJ_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++20 $(CUDA_ARCH_FLAGS) $(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -MMD -MP -c $< -o $@

$(MIXED_CUDA_OBJECT): src/backend_cuda_mixed.cu $(HEADERS) | $(OBJ_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++20 $(CUDA_ARCH_FLAGS) $(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -MMD -MP -c $< -o $@

$(BUILD_DIR)/point_vortex $(BUILD_DIR)/point_vortex_cpu: $(APP_OBJECTS) $(CPU_OBJECT) \
	$(SIMULATION_OBJECT) $(LIB_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP) -o $@ $^

$(BUILD_DIR)/point_vortex_mpi: $(APP_SOURCES) src/backend_mpi.cpp $(SIMULATION_SOURCE) \
	$(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_cuda: $(APP_OBJECTS) $(CUDA_OBJECT) $(SIMULATION_OBJECT) \
	$(LIB_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -o $@ $^

$(BUILD_DIR)/point_vortex_cuda_mixed: $(APP_OBJECTS) $(MIXED_CUDA_OBJECT) \
	$(SIMULATION_OBJECT) $(LIB_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -o $@ $^

$(BUILD_DIR)/point_vortex_initial: initial_conditions/generate_initial.cpp \
		initial_conditions/initial_condition.cpp src/read.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_tests: tests/tests.cpp $(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

test-output: $(BUILD_DIR)/point_vortex_cpu
	$(PYTHON) tests/output_integration.py $(abspath $(BUILD_DIR)/point_vortex_cpu)

test-cuda-mixed: $(BUILD_DIR)/point_vortex_cpu $(BUILD_DIR)/point_vortex_cuda_mixed
	$(PYTHON) tests/mixed_precision_consistency.py \
		$(abspath $(BUILD_DIR)/point_vortex_cpu) \
		$(abspath $(BUILD_DIR)/point_vortex_cuda_mixed)

$(BUILD_DIR)/point_vortex_audit_tests: tests/audit_tests.cpp $(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

test: $(BUILD_DIR)/point_vortex_tests $(BUILD_DIR)/point_vortex_audit_tests
	$(BUILD_DIR)/point_vortex_tests
	$(BUILD_DIR)/point_vortex_audit_tests

$(BUILD_DIR)/point_vortex_benchmark: src/benchmark.cpp src/compute.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_backend_benchmark_cpu: src/backend_benchmark.cpp src/backend_cpu.cpp \
	$(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_backend_benchmark_serial: src/backend_benchmark.cpp src/backend_cpu.cpp \
	$(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_backend_benchmark_mpi: src/backend_benchmark.cpp src/backend_mpi.cpp \
	$(LIB_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -o $@ $(filter %.cpp,$^)

$(BUILD_DIR)/point_vortex_backend_benchmark_cuda: src/backend_benchmark.cpp $(CUDA_OBJECT) \
	$(LIB_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++20 $(CUDA_ARCH_FLAGS) \
		$(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -o $@ $^

$(BUILD_DIR)/point_vortex_backend_benchmark_cuda_mixed: src/backend_benchmark.cpp \
	$(MIXED_CUDA_OBJECT) $(LIB_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++20 $(CUDA_ARCH_FLAGS) \
		$(if $(strip $(OPENMP)),-Xcompiler=$(OPENMP),) -o $@ $^

clean:
	$(RM) -r $(BUILD_DIR)

-include $(LIB_OBJECTS:.o=.d) $(SIMULATION_OBJECT:.o=.d) $(APP_OBJECTS:.o=.d) \
	$(CPU_OBJECT:.o=.d) $(CUDA_OBJECT:.o=.d) $(MIXED_CUDA_OBJECT:.o=.d)
