#include "backend.h"
#include <stdexcept>

std::unique_ptr<VelocityKernel> makeReferenceKernel(const SimParams &params) {
    if (params.boundaryCondition == "infinite")
        return std::make_unique<InfinitePlaneKernel>(params.coreRadius);
    if (params.boundaryCondition == "periodic_x")
        return std::make_unique<PeriodicXKernel>(params.boxLengthX);
    if (params.boundaryCondition == "periodic")
        return std::make_unique<PeriodicBoxKernel>(params.boxLengthX, params.boxLengthY,
                                                   params.periodicImageLayers);
    if (params.boundaryCondition == "disk")
        return std::make_unique<DiskKernel>(params.diskRadius);
    throw std::invalid_argument("unsupported boundary condition: " + params.boundaryCondition);
}
