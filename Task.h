#ifndef TASK_H
#define TASK_H

#include <complex>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class Task {
public:
    int M, N, R, id;
    int M_pad, N_pad, R_pad; // Padded dimensions
    cuComplex *d_D, *d_Q, *d_R; // Device matrices

    Task(int M, int N, int R, int id);
    ~Task();
    void padMatrixSizes();
    void setUp(); // Set up the device memory for the padded matrices
    size_t cost() const; // Cost of computation
    long int size() const; // Size of the task (matrix elements)
};

#endif // TASK_H
