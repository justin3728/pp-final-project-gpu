#include "fft_cuda.h"
#include <cstdio>
#include <cstdlib>

const int N = 1 << 20;

int main() {
    int n = N;
    int batch = 8;
    cpxcuda *in = (cpxcuda *) calloc(n * batch, sizeof(cpxcuda));
    cpxcuda *out = (cpxcuda *) calloc(n * batch, sizeof(cpxcuda));
    
    for (int i = 0; i < n * batch; i += n)
        for (int j = 0; j < n; j++) {
            in[i + j].re = j;
            in[i + j].im = 0;
        }
    
    fft_plan_cuda plan = fft_plan_cuda_1d(n, batch, in, out, false);
    fft_execute_plan_cuda(plan);
    fft_destroy_plan_cuda(plan);

    plan = fft_plan_cuda_1d(n, batch, out, in, true);
    fft_execute_plan_cuda(plan);
    fft_destroy_plan_cuda(plan);
    
    for (int i = 0; i < n * batch; i += n){
        for (int j = 0; j < 10; j++)
            printf("%d ", int(in[i + j].re + 0.5));
        puts("");
    }
    return 0;
}