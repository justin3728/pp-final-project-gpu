/*
 * File: benchmark.cpp (or benchmark_report.cpp)
 * Description: Automated Data Collection for FFT Final Report
 * Compile: make (using the Makefile provided previously)
 * Usage: ./benchmark
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>
#include "fft_cuda.h"

// 宣告在 fft_cuda.cu 中定義的控制函數
extern "C" void fft_set_tuning_block_size(int bs);

// ================= 設定區域 =================
const int WARMUP = 5;
const int REPEAT = 10; // 每個尺寸跑 10 次取平均
const int MAX_LOG_N = 20; // 測試到 2^20

// ================= 工具函數 =================
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// 計算 GFLOPS
double calc_gflops(int N, double time_ms) {
    if (time_ms == 0) return 0.0;
    // 5N log2(N)
    return (5.0 * N * log2((double)N)) / (time_ms * 1e6); // 1e6 convert ms to ns -> GFLOPS
}

// 產生隨機複數
void random_init(cpxcuda* data, int N) {
    for(int i=0; i<N; i++) {
        data[i].re = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        data[i].im = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// ================= 驗證模組 (使用 cuFFT) =================
double verify_accuracy(int N, const cpxcuda* my_result) {
    cufftDoubleComplex *d_in, *h_cufft_res;
    cpxcuda *h_in = new cpxcuda[N]; // 重生一樣的 input
    srand(12345); // 固定種子確保與 benchmark 輸入相同
    random_init(h_in, N);
    
    h_cufft_res = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*N);
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(cufftDoubleComplex)*N));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(cufftDoubleComplex)*N, cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, d_in, d_in, CUFFT_FORWARD);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cufft_res, d_in, sizeof(cufftDoubleComplex)*N, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for(int i=0; i<N; i++) {
        double diff_re = my_result[i].re - h_cufft_res[i].x;
        double diff_im = my_result[i].im - h_cufft_res[i].y;
        double err = sqrt(diff_re*diff_re + diff_im*diff_im);
        if(err > max_err) max_err = err;
    }

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_in));
    free(h_cufft_res);
    delete[] h_in;
    return max_err;
}

// ================= FFTW Baseline (CPU) =================
double benchmark_fftw(int N) {
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    // 初始化
    for(int i=0; i<N; i++) { in[i][0] = 0.5; in[i][1] = 0.5; }
    
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // 計時
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<REPEAT; i++) {
        fftw_execute(p);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    
    return (double)ms / REPEAT;
}

// ================= cuFFT Baseline (GPU) =================
double benchmark_cufft_kernel(int N) {
    cufftDoubleComplex *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(cufftDoubleComplex)*N));
    
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    
    // Warmup
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<REPEAT; i++) {
        cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    }
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_data));
    
    return (double)ms / REPEAT;
}

// ================= My CUDA Implementation =================
struct MyCudaMetrics {
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double total_ms; // E2E
    double gflops;
    double error;
};

MyCudaMetrics benchmark_my_cuda(int N) {
    cpxcuda *h_in, *h_out, *h_dummy_in;
    cudaMallocHost(&h_in, sizeof(cpxcuda)*N);
    cudaMallocHost(&h_out, sizeof(cpxcuda)*N);
    cudaMallocHost(&h_dummy_in, sizeof(cpxcuda)*N); // For plan creation
    
    srand(12345);
    random_init(h_in, N);
    
    // 1. Create Plan
    fft_plan_cuda plan = fft_plan_cuda_1d(N, 1, h_dummy_in, h_out, false);
    
    // 2. Warmup
    cudaMemcpy(plan.out_device, h_in, sizeof(cpxcuda)*N, cudaMemcpyHostToDevice);
    fft_execute_plan_cuda(plan);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 3. Measure Breakdown (H2D, Kernel, D2H)
    cudaEvent_t start_h2d, stop_h2d, start_k, stop_k, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_k);   cudaEventCreate(&stop_k);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);
    
    double sum_h2d=0, sum_k=0, sum_d2h=0;
    
    for(int i=0; i<REPEAT; i++) {
        // H2D
        cudaEventRecord(start_h2d);
        cudaMemcpy(plan.out_device, h_in, sizeof(cpxcuda)*N, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_h2d);
        
        // Kernel
        cudaEventRecord(start_k);
        fft_execute_plan_cuda(plan); // 這裡面只有 kernel launch
        cudaEventRecord(stop_k);
        
        // D2H
        cudaEventRecord(start_d2h);
        cudaMemcpy(h_out, plan.out_device, sizeof(cpxcuda)*N, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_d2h);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float t1=0, t2=0, t3=0;
        cudaEventElapsedTime(&t1, start_h2d, stop_h2d);
        cudaEventElapsedTime(&t2, start_k, stop_k);
        cudaEventElapsedTime(&t3, start_d2h, stop_d2h);
        
        sum_h2d += t1;
        sum_k   += t2;
        sum_d2h += t3;
    }
    
    MyCudaMetrics m;
    m.h2d_ms = sum_h2d / REPEAT;
    m.kernel_ms = sum_k / REPEAT;
    m.d2h_ms = sum_d2h / REPEAT;
    m.total_ms = m.h2d_ms + m.kernel_ms + m.d2h_ms;
    m.gflops = calc_gflops(N, m.kernel_ms);
    
    // 4. Verify Accuracy
    m.error = verify_accuracy(N, h_out);
    
    // Cleanup
    fft_destroy_plan_cuda(plan);
    cudaFreeHost(h_in); cudaFreeHost(h_out); cudaFreeHost(h_dummy_in);
    
    return m;
}

// ================= Main Loop =================
int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                      FFT BENCHMARK SUITE - FINAL REPORT                       ║\n");
    printf("║        Scaling Analysis | Throughput | Time Breakdown | Block Size Tuning     ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // ----------------------------------------------------------------
    // Part 1: CSV for Chart 1 (Time) & Chart 2 (Throughput)
    // ----------------------------------------------------------------
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PART 1: PERFORMANCE SCALING & THROUGHPUT ANALYSIS                              │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    printf("%-10s %-6s %-12s %-15s %-15s %-13s %-13s %-13s %-12s\n",
           "N", "Log2N", "FFTW(ms)", "MyCUDA_K(ms)", "cuFFT_K(ms)", "MyCUDA_E2E", "MyCUDA_GF", "cuFFT_GF", "Error");
    printf("─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n");
    
    // 預設 Block Size 為 256
    fft_set_tuning_block_size(256);

    for (int i = 4; i <= MAX_LOG_N; i++) {
        int N = 1 << i;
        
        // Run benchmarks
        double t_fftw = benchmark_fftw(N);
        double t_cufft = benchmark_cufft_kernel(N);
        MyCudaMetrics my_cuda = benchmark_my_cuda(N);
        
        double cufft_gflops = calc_gflops(N, t_cufft);
        
        printf("%-10d %-6d %-12.4f %-15.4f %-15.4f %-13.4f %-13.2f %-13.2f %-12.2e\n", 
               N, i, t_fftw, my_cuda.kernel_ms, t_cufft, my_cuda.total_ms, my_cuda.gflops, cufft_gflops, my_cuda.error);
        
        if (my_cuda.error > 1e-4) {
            fprintf(stderr, "⚠️  Warning: Large error detected at N=%d\n", N);
        }
    }
    
    printf("\n");

    // ----------------------------------------------------------------
    // Part 2: CSV for Chart 3 (Time Breakdown)
    // ----------------------------------------------------------------
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PART 2: TIME BREAKDOWN ANALYSIS (Host↔Device Transfer vs Kernel)               │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    printf("%-12s %-10s %-12s %-10s %-12s │ %-10s %-12s %-10s\n",
           "Size", "H2D(ms)", "Kernel(ms)", "D2H(ms)", "Total(ms)", "H2D%", "Kernel%", "D2H%");
    printf("────────────────────────────────────────────────────────────────────────────────────────────────────────\n");
    
    int breakdown_sizes[] = {1024, 1048576}; // 2^10 and 2^20
    for(int N : breakdown_sizes) {
        MyCudaMetrics m = benchmark_my_cuda(N);
        printf("N=2^%-7d %-10.4f %-12.4f %-10.4f %-12.4f │ %-10.1f%% %-12.1f%% %-10.1f%%\n",
               (int)log2(N),
               m.h2d_ms, m.kernel_ms, m.d2h_ms, m.total_ms,
               (m.h2d_ms/m.total_ms)*100, (m.kernel_ms/m.total_ms)*100, (m.d2h_ms/m.total_ms)*100);
    }
    
    printf("\n");
    
    // ----------------------------------------------------------------
    // Part 3: Chart 4 (Block Size Tuning) - AUTOMATED
    // ----------------------------------------------------------------
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PART 3: BLOCK SIZE TUNING (N = 2^20 = 1,048,576 points)                        │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    printf("%-15s %-20s %-15s\n", "Block Size", "Kernel Time (ms)", "Performance");
    printf("─────────────────────────────────────────────────────────────────────────────\n");
    
    int tuning_sizes[] = {32, 64, 128, 256, 512, 1024};
    int N_tune = 1 << 20; // 1M points
    double best_time = 1e9;
    int best_bs = 0;
    
    // First pass to find best time
    double times[6];
    for(int idx = 0; idx < 6; idx++) {
        int bs = tuning_sizes[idx];
        fft_set_tuning_block_size(bs);
        MyCudaMetrics m = benchmark_my_cuda(N_tune);
        times[idx] = m.kernel_ms;
        if(m.kernel_ms < best_time) {
            best_time = m.kernel_ms;
            best_bs = bs;
        }
    }
    
    // Second pass to display
    for(int idx = 0; idx < 6; idx++) {
        int bs = tuning_sizes[idx];
        const char* marker = (times[idx] <= best_time + 0.01) ? " ⭐ BEST" : "";
        printf("%-15d %-20.4f %s\n", bs, times[idx], marker);
    }
    
    // 恢復預設
    fft_set_tuning_block_size(256);
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           BENCHMARK COMPLETED ✓                                ║\n");
    printf("║  Optimal Block Size: %-4d │ Best Kernel Time: %.4f ms                        ║\n", best_bs, best_time);
    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    return 0;
}