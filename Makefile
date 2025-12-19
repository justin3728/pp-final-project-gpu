APP_NAME = benchmark
OBJS = fft_cuda.o

# -------------------------------------------------
# Auto-detect GPU compute capability
# -------------------------------------------------
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
GPU_CC_NODOT := $(subst .,,$(GPU_CC))

# Fallback if detection fails
ifeq ($(GPU_CC_NODOT),)
    GPU_CC_NODOT := 61
endif

# -------------------------------------------------
# Compiler settings
# -------------------------------------------------
CXX = g++
CXXFLAGS = -I. -Wall -fopenmp -std=c++11 -Wno-unknown-pragmas

# CUDA path (conda-aware)
CONDA_ENV_PATH = $(CONDA_PREFIX)
ifeq ($(CONDA_ENV_PATH),)
    CUDA_PATH = /usr/local/cuda
else
    CUDA_PATH = $(CONDA_ENV_PATH)
endif

CXXFLAGS += -I$(CUDA_PATH)/include
LDFLAGS  = -L$(CUDA_PATH)/lib -lcudart -lcufft -lfftw3

# -------------------------------------------------
# NVCC settings
# -------------------------------------------------
NVCC = nvcc
NVCCFLAGS = -O3 -m64 \
            -arch=compute_$(GPU_CC_NODOT) \
            --compiler-options '-fPIC' \
            --ptxas-options=-v

# -------------------------------------------------
# Build rules
# -------------------------------------------------
default: $(APP_NAME)

# Compile CUDA source
fft_cuda.o: fft_cuda.cu fft_cuda.h
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

# Build benchmark executable
$(APP_NAME): benchmark.cpp $(OBJS)
	$(CXX) benchmark.cpp $(OBJS) $(CXXFLAGS) $(LDFLAGS) -o $(APP_NAME)

# Clean
clean:
	rm -f *.o $(APP_NAME)
