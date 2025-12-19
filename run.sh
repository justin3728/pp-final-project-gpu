#!/bin/bash
# Quick run script for FFT benchmark

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}FFT Benchmark - Quick Run Script${NC}"
echo -e "${BLUE}================================================${NC}"

# Activate conda environment
echo -e "${GREEN}[1/3] Activating conda environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fft_cuda

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate fft_cuda environment${NC}"
    exit 1
fi

# Compile project
echo -e "${GREEN}[2/3] Compiling project...${NC}"
make clean && make

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Compilation failed${NC}"
    exit 1
fi

# Set CUDA library path
export LD_LIBRARY_PATH=/home/andy9/miniconda3/envs/fft_cuda/lib:$LD_LIBRARY_PATH

# Run benchmark and save output
echo -e "${GREEN}[3/4] Running benchmark...${NC}"
./benchmark | tee benchmark_results.txt

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${BLUE}================================================${NC}"

