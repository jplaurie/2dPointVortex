#include "backend.h"
#include <stdexcept>

std::unique_ptr<VelocityKernel> makeReferenceKernel(const SimParams &params) {
    switch (params.boundary) {
    case BoundaryKind::infinite:
        return std::make_unique<InfinitePlaneKernel>(params.coreRadius);
    case BoundaryKind::periodic_x:
        return std::make_unique<PeriodicXKernel>(params.boxLengthX);
    case BoundaryKind::periodic:
        return std::make_unique<PeriodicBoxKernel>(params.boxLengthX, params.boxLengthY,
                                                   params.periodicImageLayers);
    case BoundaryKind::disk:
        return std::make_unique<DiskKernel>(params.diskRadius);
    }
    throw std::invalid_argument("unsupported boundary condition");
}
