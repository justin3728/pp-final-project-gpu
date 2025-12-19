#ifndef FFT_CUDA_H
#define FFT_CUDA_H
#endif

struct cpxcuda {
    double re, im;
};

struct fft_plan_cuda {
    int n, batch;
    int R, C;  // Radix dimensions
    cpxcuda *out, *out_device, *ws_device, *temp_device;
    bool reverse;
    // 【優化 6】: 追蹤資料所在的緩衝區 (0=out_device, 1=temp_device)
    int result_buffer;
};

fft_plan_cuda fft_plan_cuda_1d(int n, int batch, cpxcuda *in, cpxcuda *out, bool reverse);

void fft_execute_plan_cuda(fft_plan_cuda &plan);

void fft_destroy_plan_cuda(fft_plan_cuda &plan);