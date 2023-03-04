#include <stdio.h>

// Vector addition kernel
__global__ void vectorAdd(int* a, int* b, int* c, int size) {
    int i = threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// Matrix multiplication kernel
__global__ void matrixMul(int* a, int* b, int* c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main() {
    // Vector addition and matrix multiplication setup
    const int VECTOR_SIZE = 1000000;
    const int MATRIX_SIZE = 1000;
    int* a = new int[VECTOR_SIZE];
    int* b = new int[VECTOR_SIZE];
    int* c = new int[VECTOR_SIZE];
    int* x = new int[MATRIX_SIZE * MATRIX_SIZE];
    int* y = new int[MATRIX_SIZE * MATRIX_SIZE];
    int* z = new int[MATRIX_SIZE * MATRIX_SIZE];
    int* d_a, *d_b, *d_c, *d_x, *d_y, *d_z;
    cudaMalloc(&d_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc(&d_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc(&d_c, VECTOR_SIZE * sizeof(int));
    cudaMalloc(&d_x, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_y, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_z, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = i * i;
    }
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        x[i] = i;
        y[i] = i * i;
    }
    cudaMemcpy(d_a, a, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Vector addition execution
    vectorAdd<<<1, VECTOR_SIZE>>>(d_a, d_b, d_c, VECTOR_SIZE);
    cudaMemcpy(c, d_c, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Vector addition result:\n");
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Matrix multiplication execution
    printf("\nMatrix multiplication result:\n");
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        if (i % MATRIX_SIZE == 0 && i > 0) {
            printf("\n");
        }
        printf("%d ", z[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}
