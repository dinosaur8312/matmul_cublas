#ifndef TASKPOOL_H
#define TASKPOOL_H

#include "Task.h"
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class TaskPool {
private:
    std::vector<Task> tasks;
    int numThreads;
    int nRHS;
    int matrixSize;
    cuComplex *d_inputBuffer, *d_outputBuffer, *d_internalBuffer, *d_matmat;
    std::vector<cudaStream_t> streams;
    std::vector<cublasHandle_t> cublasHandles;

public:
    TaskPool(const std::string &MILFile, int numThreads, int nRHS, const std::string &filter);
    ~TaskPool();
    void processTasks(int numRepeat = 1);
    void prepareBuffers();
    void freeBuffers();
    void loadTasks(const std::string &MILFile, const std::string &filter);
    //std::vector<std::string> SplitStringOnWhiteSpace(const std::string &input);
    void addTask(const Task &task);
};

#endif // TASKPOOL_H
