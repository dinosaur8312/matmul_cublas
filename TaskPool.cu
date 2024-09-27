#include "TaskPool.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <set>
#include <sstream>
#include <algorithm> // For std::sort

// Utility function to split strings by whitespace
std::vector<std::string> SplitStringOnWhiteSpace(const std::string &input) {
    std::vector<std::string> result;
    std::istringstream istr(input);
    while (istr) {
        std::string data;
        istr >> data;
        if (!data.empty())
            result.push_back(data);
    }
    return result;
}

// Constructor to read tasks from MIL file and initialize CUDA streams and cuBLAS handles
TaskPool::TaskPool(const std::string &MILFile, int numThreads, int nRHS, const std::string &filter)
    : numThreads(numThreads), nRHS(nRHS), matrixSize(0) {

    streams.resize(numThreads);
    cublasHandles.resize(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        cublasCreate(&cublasHandles[i]);
        cublasSetStream(cublasHandles[i], streams[i]);
    }

    loadTasks(MILFile, filter);
    printf("Loaded %lu tasks from MIL file\n", tasks.size());  // Debug print

    prepareBuffers();
    printf("Prepared buffers for task processing\n");  // Debug print
}

// Destructor to free device memory, destroy CUDA streams and cuBLAS handles
TaskPool::~TaskPool() {
    for (int i = 0; i < numThreads; ++i) {
        cublasDestroy(cublasHandles[i]);
        cudaStreamDestroy(streams[i]);
    }
    freeBuffers();
}

// Add task to the vector
void TaskPool::addTask(const Task &task) {
    tasks.push_back(task);
}

// Load tasks from MIL file
void TaskPool::loadTasks(const std::string &MILFile, const std::string &filter) {
    std::fstream fMil;
    fMil.open(MILFile, std::fstream::in);
    std::string thisLine = "";
    std::set<int> includedMILs;
    long long int numElements = 0;

    while (!fMil.eof()) {
        std::getline(fMil, thisLine);
        std::vector<std::string> words = SplitStringOnWhiteSpace(thisLine);
        printf("Read line: %s\n", thisLine.c_str());  // Debug print
        printf("Number of words: %lu\n", words.size());  // Debug print
        printf("fMil.eof() = %d\n", fMil.eof());  // Debug print

        if (filter != "")
        {
            //print filter
            if (words.size() < 7)
            {
                std::cout << "Words size: " << words.size() << std::endl;
                continue;
            }
            if (filter != words[3])
            {
                std::cout << "Filter: " << filter << std::endl;  
                std::cout << "Words[3]: " << words[3] << std::endl;          
                continue;
            }
        }
        int M = std::atoi(words[4].c_str());
        int N = std::atoi(words[5].c_str());
        int R = std::atoi(words[6].c_str());
        printf("Read task with M: %d, N: %d, R: %d\n", M, N, R);  // Debug print

        if (includedMILs.find(std::atoi(words[2].c_str())) == includedMILs.end())
            includedMILs.insert(std::atoi(words[2].c_str()));
        else
            continue;

        Task task(M, N, R, tasks.size());
        addTask(task);
        printf("Added task %lu with M: %d, N: %d, R: %d\n", tasks.size() - 1, M, N, R);  // Debug print
        numElements += task.size();
        printf("Current number of elements: %lld\n", numElements);  // Debug print
    }

    fMil.close();
    printf("Total number of elements: %lld\n", numElements);  // Debug print
    matrixSize = static_cast<int>(ceil(sqrt(1.0 * numElements)));

    printf("Matrix size: %d\n", matrixSize);  // Debug print

    // Sort tasks based on the computation cost (to optimize execution)
    //std::sort(tasks.begin(), tasks.end(), [](const Task &a, const Task &b) {
    //    return a.cost() < b.cost();
    //});
}

// Allocate buffers in device memory
void TaskPool::prepareBuffers() {
    int maxSrcBuffer = 0, maxTestBuffer = 0, maxRankBuffer = 0;

    for (const Task &task : tasks) {
        maxSrcBuffer = std::max(maxSrcBuffer, task.N_pad);
        maxTestBuffer = std::max(maxTestBuffer, task.M_pad);
        maxRankBuffer = std::max(maxRankBuffer, task.R_pad);
    }

    cudaMalloc(&d_inputBuffer, nRHS * maxSrcBuffer * sizeof(cuComplex));
    cudaMalloc(&d_outputBuffer, nRHS * maxTestBuffer * sizeof(cuComplex));
    cudaMalloc(&d_internalBuffer, nRHS * maxRankBuffer * sizeof(cuComplex));
    cudaMalloc(&d_matmat, nRHS * matrixSize * sizeof(cuComplex));
    cudaMemset(d_matmat, 0, nRHS * matrixSize * sizeof(cuComplex));
}

void TaskPool::freeBuffers() {
    cudaFree(d_inputBuffer);
    cudaFree(d_outputBuffer);
    cudaFree(d_internalBuffer);
    cudaFree(d_matmat);
}

// Process tasks using CUDA streams and cuBLAS for matrix multiplication
// Process tasks using CUDA streams and cuBLAS for matrix multiplication
void TaskPool::processTasks(int numRepeat) {
    double totalCost = 0.0;
    for (const Task &task : tasks) {
        totalCost += task.cost();
    }
    totalCost *= numRepeat * nRHS / 1e9;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    printf("Starting task processing...\n");  // Debug print

    for (int iter = 0; iter < numRepeat; ++iter) {
        printf("Iteration %d of %d\n", iter + 1, numRepeat);  // Debug print

        for (size_t i = 0; i < tasks.size(); ++i) {
            Task &task = tasks[i];
            int streamIndex = i % numThreads;  // Assign tasks round-robin to streams

            printf("Processing task %lu, M: %d, N: %d, R: %d\n", i, task.M, task.N, task.R);  // Debug print
            printf("Task %lu is assigned to stream %d\n", i, streamIndex);  // Debug print

            cuComplex *d_B = d_inputBuffer;
            cuComplex *d_C = d_outputBuffer;
            cuComplex *d_local = d_internalBuffer;

            printf("Buffers assigned: d_B, d_C, d_local\n");  // Debug print

            const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
            const cuComplex beta = make_cuComplex(0.0f, 0.0f);

            if (task.R == 0) {
                // Single matrix multiplication
                printf("Performing single matrix multiplication for task %lu\n", i);
                cublasStatus_t status = cublasCgemm(cublasHandles[streamIndex], CUBLAS_OP_N, CUBLAS_OP_N,
                                                    task.M_pad, nRHS, task.N_pad,
                                                    &alpha, task.d_D, task.M_pad,
                                                    d_B, task.N_pad, &beta,
                                                    d_C, task.M_pad);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("cublasCgemm failed for task %lu with status %d\n", i, status);
                }
            } else {
                // Two-stage matrix multiplication
                printf("Performing two-stage matrix multiplication for task %lu\n", i);
                cublasStatus_t status_R = cublasCgemm(cublasHandles[streamIndex], CUBLAS_OP_N, CUBLAS_OP_N,
                                                      task.R_pad, nRHS, task.N_pad,
                                                      &alpha, task.d_R, task.R_pad,
                                                      d_B, task.N_pad, &beta,
                                                      d_local, task.R_pad);
                if (status_R != CUBLAS_STATUS_SUCCESS) {
                    printf("First cublasCgemm (R stage) failed for task %lu with status %d\n", i, status_R);
                }

                cublasStatus_t status_Q = cublasCgemm(cublasHandles[streamIndex], CUBLAS_OP_N, CUBLAS_OP_N,
                                                      task.M_pad, nRHS, task.R_pad,
                                                      &alpha, task.d_Q, task.M_pad,
                                                      d_local, task.R_pad, &beta,
                                                      d_C, task.M_pad);
                if (status_Q != CUBLAS_STATUS_SUCCESS) {
                    printf("Second cublasCgemm (Q stage) failed for task %lu with status %d\n", i, status_Q);
                }
            }
            printf("Completed task %lu on stream %d\n", i, streamIndex);  // Debug print
        }
    }

    // Synchronize all streams to ensure all tasks are done
    for (int i = 0; i < numThreads; ++i) {
        printf("Synchronizing stream %d\n", i);  // Debug print
        cudaStreamSynchronize(streams[i]);
    }

    // Record stop time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Total elapsed time in seconds
    float elapsedTimeInSeconds = milliseconds / 1000.0f;

    // Calculate GFLOPS
    double gflops = totalCost / elapsedTimeInSeconds;

    // Output results
    printf("Total elapsed time: %f seconds\n", elapsedTimeInSeconds);
    printf("Performance: %f GFLOP/S\n", gflops);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

