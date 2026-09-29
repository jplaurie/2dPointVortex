#include "backend.h"
#include "read.h"
#include "simulation.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    int exitCode = 0;
    bool backendInitialized = false;
    try {
        backendInitialize(argc, argv);
        backendInitialized = true;
        if (argc > 2)
            throw std::invalid_argument("expected at most one parameter file argument");

        const std::string parameterFile = argc > 1 ? argv[1] : "params.txt";
        SimParams params = loadParams(parameterFile);
#ifdef _OPENMP
        if (params.numThreads > 0)
            omp_set_num_threads(params.numThreads);
#endif
        runSimulation(std::move(params), parameterFile);
    } catch (const std::exception &error) {
        if (backendIsRoot())
            std::cerr << "error: " << error.what() << '\n';
        exitCode = 1;
        if (backendInitialized)
            backendAbort(exitCode);
    }
    if (backendInitialized)
        backendFinalize();
    return exitCode;
}
