#ifndef POINT_VORTEX_INITIAL_CONDITION_H
#define POINT_VORTEX_INITIAL_CONDITION_H

#include "vortex.h"
#include <cstddef>
#include <cstdint>
#include <string>

enum class InitialGeometry { infinite, periodic_x, periodic, disk };
enum class InitialPattern { random, single, corotatingPair, dipole, ring };

struct InitialConditionOptions {
    InitialGeometry geometry = InitialGeometry::infinite;
    InitialPattern pattern = InitialPattern::random;
    std::size_t count = 100;
    std::uint64_t seed = 1234567;
    double minimumSeparation = 0.0;
    double circulationMagnitude = 1.0;
    double infiniteHalfWidth = 1.0;
    double boxLength = 2.0;
    double diskRadius = 1.0;
    double ringRadius = 0.0; // Zero selects a geometry-aware default.
};

VortexSystem generateInitialCondition(const InitialConditionOptions &options);
void validateInitialCondition(const VortexSystem &vortices, const InitialConditionOptions &options);
double minimumVortexSeparation(const VortexSystem &vortices,
                               const InitialConditionOptions &options);
void writeInitialCondition(const std::string &filename, const VortexSystem &vortices,
                           const InitialConditionOptions &options, bool overwrite = false);

const char *toString(InitialGeometry geometry);
const char *toString(InitialPattern pattern);

#endif
