#include "TaskPool.h"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <MILFile> <numThreads> <numRepeat> <numRHS> <filter>" << std::endl;
        return 1;
    }

    std::string MILFile = argv[1];
    int numThreads = std::atoi(argv[2]);
    int numRepeat = std::atoi(argv[3]);
    int numRHS = std::atoi(argv[4]);
    std::string filter = argv[5];

    TaskPool taskPool(MILFile, numThreads, numRHS, filter);

    // Process all tasks in the task pool using CUDA streams concurrently
    taskPool.processTasks(numRepeat);

    return 0;
}
