#ifndef POINT_VORTEX_CHECKPOINT_H
#define POINT_VORTEX_CHECKPOINT_H
#include "compute.h"
#include "dipole.h"
#include "params.h"
#include "vortex.h"
#include <cstddef>
#include <filesystem>
struct OutputSchedule {
    double trajectoryInterval = 0.0;
    double diagnosticsInterval = 0.0;
    double checkpointInterval = 0.0;
    double nextDiagnosticsTime = 0.0;
    double nextCheckpointTime = 0.0;
};
struct CheckpointProgress {
    double time;
    double suggestedTimeStep;
    double nextOutputTime;
    std::size_t acceptedSteps;
    std::size_t outputIndex;
    std::size_t eventIndex;
    double nextDipoleRemovalTime = 0.0;
};
struct Checkpoint {
    // Restart state includes output/integrator progress as well as vortex data.
    VortexSystem vortices;
    Invariants initialInvariants;
    Invariants segmentInvariants;
    bool hasSegmentInvariants = false;
    double time = 0.0;
    double suggestedTimeStep = 0.0;
    double nextOutputTime = 0.0;
    bool hasOutputSchedule = false;
    OutputSchedule outputSchedule;
    std::size_t acceptedSteps = 0;
    std::size_t outputIndex = 0;
    std::size_t eventIndex = 0;
    double coreRadius = 0.0;
    IntegratorKind integrator = IntegratorKind::dopri5;
    std::string boundaryCondition = "infinite";
    double geometryLengthX = 0.0;
    double geometryLengthY = 0.0;
    int periodicImageLayers = 0;
    bool dipoleRemoval = false;
    double dipoleRemovalDistance = 0.0;
    bool hasDipoleUpperConfig = false;
    bool dipoleRemovalUpper = false;
    double dipoleRemovalUpperDistance = 0.0;
    bool hasDipoleSchedule = false;
    double dipoleRemovalInterval = 0.0;
    double nextDipoleRemovalTime = 0.0;
    ReinjectionMode dipoleReinjection = ReinjectionMode::none;
    DipoleEventState dipoleState;
};
Checkpoint loadCheckpoint(const std::filesystem::path &filename);
void writeCheckpoint(const std::filesystem::path &directory, const VortexSystem &vortices,
                     const Invariants &initialInvariants, const SimParams &params,
                     const Invariants &segmentInvariants, const DipoleEventState &dipoleState,
                     const OutputSchedule &outputSchedule, const CheckpointProgress &progress,
                     bool overwrite = false);
std::filesystem::path checkpointPath(const std::filesystem::path &directory,
                                     std::size_t outputIndex);
#endif
