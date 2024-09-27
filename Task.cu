#include "Task.h"
#include <cuda_runtime.h>
#include <iostream>

// Constructor: Initialize task with matrix sizes and allocate CUDA device memory
Task::Task(int M, int N, int R, int id) : M(M), N(N), R(R), id(id), d_D(nullptr), d_Q(nullptr), d_R(nullptr) {
    padMatrixSizes();  // Set padded sizes
    setUp();           // Allocate device memory based on padded sizes
}

// Destructor: Free CUDA device memory
Task::~Task() {
    if (d_D) {
        cudaFree(d_D);
    }
    if (d_Q) {
        cudaFree(d_Q);
    }
    if (d_R) {
        cudaFree(d_R);
    }
}

// Pad matrix dimensions to powers of 2, with minimum of 128
void Task::padMatrixSizes() {
    M_pad = (M < 128) ? 128 : 1 << (int)std::ceil(std::log2(M));
    N_pad = (N < 128) ? 128 : 1 << (int)std::ceil(std::log2(N));
    R_pad = (R < 128 && R != 0) ? 128 : (R == 0) ? 0 : 1 << (int)std::ceil(std::log2(R));
}

// Allocate device memory for padded matrices
void Task::setUp() {
    if (R == 0) {
        // Allocate memory for dense matrix D if R == 0
        cudaMalloc(&d_D, M_pad * N_pad * sizeof(cuComplex));
        std::cout << "Allocated dense matrix of size " << M_pad << " x " << N_pad << " on device for Task " << id << std::endl;
    } else {
        // Allocate memory for Q and R matrices if R != 0
        cudaMalloc(&d_Q, M_pad * R_pad * sizeof(cuComplex));
        cudaMalloc(&d_R, R_pad * N_pad * sizeof(cuComplex));
        std::cout << "Allocated Q matrix of size " << M_pad << " x " << R_pad << " on device for Task " << id << std::endl;
        std::cout << "Allocated R matrix of size " << R_pad << " x " << N_pad << " on device for Task " << id << std::endl;
    }
}

size_t Task::cost() const {
    return (R_pad!=0) ? M_pad * R_pad + R_pad * N_pad : M_pad * N_pad;
}

long int Task::size() const {
    return M * N;
}
