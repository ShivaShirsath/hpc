#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024 // Vector and matrix size

__global__ void vectorAdd(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

__global__ void matrixMul(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += a[row * N + k] * b[k * N + col];
        c[row * N + col] = sum;
    }
}

int main()
{
    int *a, *b, *c, *d, *e, *f;
    cudaMallocManaged(&a, N * N * sizeof(int));
    cudaMallocManaged(&b, N * N * sizeof(int));
    cudaMallocManaged(&c, N * N * sizeof(int));
    cudaMallocManaged(&d, N * sizeof(int));
    cudaMallocManaged(&e, N * sizeof(int));
    cudaMallocManaged(&f, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        d[i] = i;
        e[i] = N - i;
    }
    for (int i = 0; i < N * N; i++)
    {
        a[i] = i;
        b[i] = N * N - i;
    }
    vectorAdd<<<(N + 255) / 256, 256>>>(d, e, f);
    matrixMul<<<dim3((N + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(a, b, c);
    cudaDeviceSynchronize();
    printf("Vector addition result:\n");
    for (int i = 0; i < N; i++)
        printf("%d ", f[i]);
    printf("\n");
    printf("Matrix multiplication result:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d ", c[i * N + j]);
        printf("\n");
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);
    return 0;
}
