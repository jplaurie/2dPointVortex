#include "backend.h"
#include "checkpoint.h"
#include "compute.h"
#include "initial_condition.h"
#include "print.h"
#include "read.h"
#include "timestep.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

bool sameFile(const std::filesystem::path &left, const std::filesystem::path &right) {
    return std::filesystem::weakly_canonical(left) == std::filesystem::weakly_canonical(right) ||
           (std::filesystem::exists(left) && std::filesystem::exists(right) &&
            std::filesystem::equivalent(left, right));
}

bool hasManagedRunOutput(const RunPaths &paths) {
    return std::filesystem::exists(paths.trajectory) ||
           std::filesystem::exists(paths.diagnostics) ||
           std::filesystem::exists(paths.checkpoints) ||
           std::filesystem::exists(paths.directory / "resolved_parameters.txt") ||
           std::filesystem::exists(paths.directory / "segments");
}

void removeManagedRunOutput(const RunPaths &paths) {
    std::error_code error;
    for (const auto &path :
         {paths.trajectory, paths.diagnostics, paths.directory / "resolved_parameters.txt"}) {
        std::filesystem::remove(path, error);
        if (error)
            throw std::runtime_error("cannot remove previous run output: " + path.string() + ": " +
                                     error.message());
    }
    for (const auto &path : {paths.checkpoints, paths.directory / "segments"}) {
        std::filesystem::remove_all(path, error);
        if (error)
            throw std::runtime_error("cannot remove previous run output: " + path.string() + ": " +
                                     error.message());
    }
}

struct GeometrySignature {
    double lengthX;
    double lengthY;
    int imageLayers;
};

GeometrySignature geometrySignature(const SimParams &params) {
    switch (params.boundary) {
    case BoundaryKind::periodic_x:
        return {params.boxLengthX, 0.0, 0};
    case BoundaryKind::periodic:
        return {params.boxLengthX, params.boxLengthY, params.periodicImageLayers};
    case BoundaryKind::disk:
        return {params.diskRadius, 0.0, 0};
    case BoundaryKind::infinite:
        return {0.0, 0.0, 0};
    }
    throw std::logic_error("unsupported boundary kind");
}

bool checkpointMatches(const Checkpoint &checkpoint, const SimParams &params) {
    const GeometrySignature geometry = geometrySignature(params);
    if (checkpoint.coreRadius != params.coreRadius || checkpoint.integrator != params.integrator ||
        checkpoint.boundary != params.boundary ||
        checkpoint.geometryLengthX != geometry.lengthX ||
        checkpoint.geometryLengthY != geometry.lengthY ||
        checkpoint.periodicImageLayers != geometry.imageLayers ||
        checkpoint.dipoleRemoval != params.dipoleRemoval) {
        return false;
    }

    if (!params.dipoleRemoval)
        return true;

    const bool upperRemovalMatches =
        (!checkpoint.hasDipoleUpperConfig && !params.dipoleRemovalUpper) ||
        (checkpoint.hasDipoleUpperConfig &&
         checkpoint.dipoleRemovalUpper == params.dipoleRemovalUpper &&
         (!params.dipoleRemovalUpper || checkpoint.dipoleRemovalUpperDistance ==
                                              params.dipoleRemovalUpperDistance));
    const bool scheduleMatches =
        !checkpoint.hasDipoleSchedule ||
        checkpoint.dipoleRemovalInterval == params.dipoleRemovalInterval;

    return checkpoint.dipoleRemovalDistance == params.dipoleRemovalDistance &&
           upperRemovalMatches && scheduleMatches &&
           checkpoint.dipoleReinjection == params.dipoleReinjection;
}

InitialGeometry initialGeometry(BoundaryKind boundary) {
    switch (boundary) {
    case BoundaryKind::infinite:
        return InitialGeometry::infinite;
    case BoundaryKind::periodic_x:
        return InitialGeometry::periodic_x;
    case BoundaryKind::periodic:
        return InitialGeometry::periodic;
    case BoundaryKind::disk:
        return InitialGeometry::disk;
    }
    throw std::logic_error("unsupported boundary kind");
}

InitialConditionOptions initialConditionOptions(const SimParams &params) {
    InitialConditionOptions options;
    options.geometry = initialGeometry(params.boundary);
    options.count = std::max<std::size_t>(1, params.vortexCount);
    options.seed = params.randomSeed;
    options.boxLength = params.boxLengthX;
    options.diskRadius = params.diskRadius;
    return options;
}

bool significantlyDifferent(double left, double right) {
    return std::abs(left - right) > 1e-13 * std::max(left, right);
}

void validateInitialConditionMetadata(const InitialConditionMetadata &metadata,
                                      const SimParams &params) {
    if (metadata.geometry && *metadata.geometry != toString(params.boundary)) {
        throw std::invalid_argument("initial-condition geometry is " + *metadata.geometry +
                                    " but boundaryCondition is " + toString(params.boundary));
    }

    const bool periodicX = params.boundary == BoundaryKind::periodic ||
                           params.boundary == BoundaryKind::periodic_x;
    if (periodicX && metadata.boxLength) {
        const bool xMismatch = significantlyDifferent(*metadata.boxLength, params.boxLengthX);
        const bool yMismatch = params.boundary == BoundaryKind::periodic &&
                               significantlyDifferent(*metadata.boxLength, params.boxLengthY);
        if (xMismatch || yMismatch) {
            throw std::invalid_argument(
                "initial-condition periodic length does not match simulation parameters");
        }
    }

    if (params.boundary == BoundaryKind::disk && metadata.diskRadius &&
        significantlyDifferent(*metadata.diskRadius, params.diskRadius)) {
        throw std::invalid_argument(
            "initial-condition disk radius does not match simulation parameters");
    }
}

VortexSystem loadInitialState(const SimParams &params) {
    const InitialConditionMetadata metadata =
        readInitialConditionMetadata(params.initialConditionFile);
    validateInitialConditionMetadata(metadata, params);

    VortexSystem vortices = loadVortices(params.initialConditionFile);
    InitialConditionOptions options = initialConditionOptions(params);
    options.count = vortices.size();
    validateInitialCondition(vortices, options);
    return vortices;
}

InitialPattern initialPattern(InitialConditionKind kind) {
    switch (kind) {
    case InitialConditionKind::random:
        return InitialPattern::random;
    case InitialConditionKind::ring:
        return InitialPattern::ring;
    case InitialConditionKind::single:
        return InitialPattern::single;
    case InitialConditionKind::dipole:
        return InitialPattern::dipole;
    case InitialConditionKind::file:
        throw std::logic_error("file initial condition was not loaded");
    }
    throw std::logic_error("unsupported initial condition");
}

VortexSystem makeInitialState(const SimParams &params) {
    if (params.initialCondition == InitialConditionKind::file)
        return loadInitialState(params);

    InitialConditionOptions options = initialConditionOptions(params);
    options.pattern = initialPattern(params.initialCondition);

    // Preserve the historical unit-radius ring in the infinite plane. Other
    // geometries use the generator's boundary-aware default radius.
    if (params.initialCondition == InitialConditionKind::ring &&
        params.boundary == BoundaryKind::infinite)
        options.ringRadius = 1.0;
    return generateInitialCondition(options);
}

double nextScheduledTime(bool restarting, double currentTime, double interval,
                         double savedInterval, double savedNextTime) {
    const double nextTime =
        restarting && interval == savedInterval ? savedNextTime : currentTime + interval;
    if (!std::isfinite(nextTime) || !(nextTime > currentTime))
        throw std::runtime_error("output interval cannot advance simulation time");
    return nextTime;
}

double startingTimeStep(const SimParams &params, const Checkpoint &restart, bool restarting) {
    if (restarting)
        return restart.suggestedTimeStep;
    if (params.integrator == IntegratorKind::rk4)
        return params.timeStep;
    return std::clamp(params.timeStep, params.minimumTimeStep, params.maximumTimeStep);
}

struct ScheduleState {
    double nextTrajectoryTime;
    OutputSchedule output;
    double nextDipoleRemovalTime;
};

ScheduleState makeSchedule(const SimParams &params, const Checkpoint &restart, bool restarting,
                           double currentTime) {
    // Preserve a saved cadence when its interval is unchanged. Changing an
    // interval deliberately starts a new cadence at the restart time.
    const OutputSchedule &saved = restart.outputSchedule;
    const bool hasSavedSchedule = restarting && restart.hasOutputSchedule;
    const double nextTrajectoryTime = nextScheduledTime(
        restarting, currentTime, params.outputTime,
        hasSavedSchedule ? saved.trajectoryInterval : params.outputTime, restart.nextOutputTime);

    OutputSchedule output{
        params.outputTime,
        params.diagnosticsInterval(),
        params.checkpointInterval(),
        nextScheduledTime(restarting, currentTime, params.diagnosticsInterval(),
                          hasSavedSchedule ? saved.diagnosticsInterval : params.outputTime,
                          hasSavedSchedule ? saved.nextDiagnosticsTime : restart.nextOutputTime),
        nextScheduledTime(restarting, currentTime, params.checkpointInterval(),
                          hasSavedSchedule ? saved.checkpointInterval : params.outputTime,
                          hasSavedSchedule ? saved.nextCheckpointTime : restart.nextOutputTime)};

    double nextDipoleRemovalTime = 0.0;
    if (params.dipoleRemoval && params.dipoleRemovalInterval > 0.0) {
        const bool preserveDipoleSchedule =
            restarting && restart.hasDipoleSchedule &&
            restart.dipoleRemovalInterval == params.dipoleRemovalInterval;
        nextDipoleRemovalTime = preserveDipoleSchedule
                                    ? restart.nextDipoleRemovalTime
                                    : currentTime + params.dipoleRemovalInterval;
        if (!std::isfinite(nextDipoleRemovalTime) ||
            !(nextDipoleRemovalTime > currentTime)) {
            throw std::runtime_error("dipole-removal interval cannot advance simulation time");
        }
    }

    return {nextTrajectoryTime, output, nextDipoleRemovalTime};
}

void protectInputFiles(const std::filesystem::path &destination,
                       const std::string &parameterFile, const SimParams &params) {
    for (const std::string &input :
         {parameterFile, params.initialConditionFile, params.restartFile}) {
        if (!input.empty() && sameFile(destination, input))
            throw std::runtime_error("output path would overwrite input file: " + input);
    }
}

void validateCheckpointDestination(const RunPaths &paths, std::size_t index,
                                   const std::string &parameterFile,
                                   const SimParams &params) {
    const std::filesystem::path destination = checkpointPath(paths.checkpoints, index);
    protectInputFiles(destination, parameterFile, params);
    if (sameFile(destination, paths.trajectory) || sameFile(destination, paths.diagnostics))
        throw std::runtime_error("checkpoint and CSV output paths must be different");
}

struct RunWriters {
    std::unique_ptr<TrajectoryWriter> trajectory;
    std::unique_ptr<DiagnosticsWriter> diagnostics;
};

RunWriters openRunWriters(const RunPaths &paths, const SimParams &params,
                          const std::string &parameterFile, const Invariants &initial,
                          const OutputSchedule &schedule, std::size_t firstCheckpointIndex) {
    // Check every managed destination before replacing any existing artefact.
    const std::filesystem::path trajectoryPath =
        std::filesystem::weakly_canonical(paths.trajectory);
    const std::filesystem::path diagnosticsPath =
        std::filesystem::weakly_canonical(paths.diagnostics);
    if (sameFile(trajectoryPath, diagnosticsPath))
        throw std::runtime_error("managed trajectory and diagnostics paths must be different");

    protectInputFiles(trajectoryPath, parameterFile, params);
    protectInputFiles(diagnosticsPath, parameterFile, params);
    validateCheckpointDestination(paths, firstCheckpointIndex, parameterFile, params);

    if (hasManagedRunOutput(paths)) {
        if (!params.overwriteRun) {
            throw std::runtime_error("run directory already contains solver output: " +
                                     paths.directory.string() +
                                     " (choose another runDirectory or set overwriteRun true)");
        }
        removeManagedRunOutput(paths);
    }

    RunWriters writers{
        std::make_unique<TrajectoryWriter>(paths.trajectory.string(), false),
        std::make_unique<DiagnosticsWriter>(paths.diagnostics.string(), initial, false)};
    std::cout << "backend=" << backendName() << '\n'
              << "trajectory=" << std::filesystem::absolute(paths.trajectory).lexically_normal()
              << " interval=" << schedule.trajectoryInterval << '\n'
              << "diagnostics=" << std::filesystem::absolute(paths.diagnostics).lexically_normal()
              << " interval=" << schedule.diagnosticsInterval << '\n'
              << "checkpoints=" << std::filesystem::absolute(paths.checkpoints).lexically_normal()
              << " interval=" << schedule.checkpointInterval << '\n'
              << std::flush;
    return writers;
}

} // namespace

int main(int argc, char **argv) {
    int exitCode = 0;
    bool backendInitialized = false;
    try {
        backendInitialize(argc, argv);
        backendInitialized = true;
        if (argc > 2)
            throw std::invalid_argument("expected at most one parameter file argument");
        const std::string parameterFile = argc > 1 ? argv[1] : "params.txt";
        const SimParams params = loadParams(parameterFile);
        const RunPaths paths = runPaths(params);

#ifdef _OPENMP
        if (params.numThreads > 0)
            omp_set_num_threads(params.numThreads);
#endif

        const bool restarting = !params.restartFile.empty();
        Checkpoint restart;
        if (restarting)
            restart = loadCheckpoint(params.restartFile);

        if (restarting && !checkpointMatches(restart, params))
            throw std::runtime_error(
                "checkpoint geometry or integrator settings do not match parameters");

        VortexSystem vortices = restarting ? std::move(restart.vortices) : makeInitialState(params);
        auto kernel = makeBackendKernel(params);
        const Invariants initial =
            restarting ? restart.initialInvariants : computeInvariants(vortices, *kernel);
        DipoleManager dipoles =
            restarting ? DipoleManager(params, restart.dipoleState) : DipoleManager(params);
        if (!restarting)
            dipoles.process(vortices);
        Invariants segmentReference = restarting && restart.hasSegmentInvariants
                                          ? restart.segmentInvariants
                                          : computeInvariants(vortices, *kernel);
        RungeKuttaIntegrator integrator(vortices.size());
        VelocityField velocity(vortices.size());
        const bool deviceStepping = kernel->supportsDeviceStepping();
        bool hostStateCurrent = true;
        if (deviceStepping)
            kernel->uploadDeviceState(vortices);

        double time = restarting ? restart.time : 0.0;
        if (time > params.endTime)
            throw std::runtime_error("checkpoint time is later than endTime");
        double dt = startingTimeStep(params, restart, restarting);
        const ScheduleState initialSchedule = makeSchedule(params, restart, restarting, time);
        double nextOutput = initialSchedule.nextTrajectoryTime;
        OutputSchedule schedule = initialSchedule.output;
        double nextDipoleRemoval = initialSchedule.nextDipoleRemovalTime;
        std::size_t acceptedSteps = restarting ? restart.acceptedSteps : 0;
        // The filename index counts checkpoints, independently of CSV frames.
        std::size_t outputIndex = restarting ? restart.outputIndex : 0;
        // One event frame identifies a simultaneous trajectory/diagnostics/checkpoint save.
        std::size_t eventIndex = restarting ? restart.eventIndex : 0;

        // Detect the common rerun/restart collision before opening and possibly truncating CSVs.
        if (restarting && restart.outputIndex == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("checkpoint index overflow");
        const std::size_t firstCheckpointIndex = restarting ? restart.outputIndex + 1 : 0;
        std::unique_ptr<TrajectoryWriter> trajectory;
        std::unique_ptr<DiagnosticsWriter> diagnostics;
        if (backendIsRoot()) {
            RunWriters writers = openRunWriters(paths, params, parameterFile, initial, schedule,
                                                firstCheckpointIndex);
            trajectory = std::move(writers.trajectory);
            diagnostics = std::move(writers.diagnostics);
        }

        const auto synchronizeDeviceState = [&] {
            if (deviceStepping && !hostStateCurrent) {
                kernel->downloadDeviceState(vortices);
                hostStateCurrent = true;
            }
        };
        const auto writeTrajectory = [&] {
            synchronizeDeviceState();
            if (deviceStepping)
                kernel->evaluateDeviceState(velocity);
            else
                kernel->evaluate(vortices, velocity);
            if (backendIsRoot())
                trajectory->write(time, eventIndex, vortices, velocity);
        };
        const auto writeDiagnostics = [&] {
            synchronizeDeviceState();
            const Invariants current = computeInvariants(vortices, *kernel);
            const DipoleEventState eventState = dipoles.state();
            if (backendIsRoot()) {
                diagnostics->write(time, eventIndex, current, segmentReference,
                                   eventState.removedPairs, eventState.removedUpperPairs,
                                   eventState.reinjectedPairs);
                printDiagnostics(time, acceptedSteps, current, initial, params.boundary,
                                 segmentReference, eventState.removedPairs,
                                 eventState.removedUpperPairs,
                                 eventState.reinjectedPairs);
            }
        };
        const auto writeCurrentCheckpoint = [&] {
            synchronizeDeviceState();
            if (backendIsRoot()) {
                validateCheckpointDestination(paths, outputIndex, parameterFile, params);
                writeCheckpoint(paths.checkpoints, vortices, initial, params, segmentReference,
                                dipoles.state(), schedule,
                                {time, dt, nextOutput, acceptedSteps, outputIndex, eventIndex,
                                 nextDipoleRemoval});
            }
        };

        // A restarted branch records its starting frame but does not duplicate its source
        // checkpoint.
        writeTrajectory();
        writeDiagnostics();
        if (!restarting)
            writeCurrentCheckpoint();
        if (backendIsRoot())
            writeRunProvenance(params, parameterFile, backendName(), backendRuntimeDetails(), time,
                               eventIndex, restarting);

        const auto takeStep = [&](double stepSize) {
            StepResult step{stepSize, dt, 0.0, 0};
            if (deviceStepping) {
                if (params.integrator == IntegratorKind::rk4)
                    kernel->deviceRk4Step(stepSize);
                else
                    step = kernel->deviceDopri5Step(
                        stepSize, params.absoluteTolerance, params.relativeTolerance,
                        params.minimumTimeStep, params.maximumTimeStep);
                hostStateCurrent = false;
            } else if (params.integrator == IntegratorKind::rk4) {
                integrator.rk4Step(vortices, stepSize, *kernel);
            } else {
                step = integrator.dopri5Step(vortices, stepSize, *kernel, params);
            }
            return step;
        };
        const auto advanceClock = [&](double &next, double interval) {
            do {
                const double following = next + interval;
                if (!std::isfinite(following) || !(following > next))
                    throw std::runtime_error(
                        "scheduled interval cannot advance simulation time");
                next = following;
            } while (next <= time);
        };

        while (time < params.endTime) {
            // Land on the next removal/output event, or the final time.
            const double timeToDipoleRemoval =
                params.dipoleRemoval && params.dipoleRemovalInterval > 0.0
                    ? nextDipoleRemoval - time
                    : std::numeric_limits<double>::infinity();
            const double stepSize =
                std::min({dt, params.endTime - time, nextOutput - time,
                          schedule.nextDiagnosticsTime - time,
                          schedule.nextCheckpointTime - time, timeToDipoleRemoval});
            if (!(time + stepSize > time))
                throw std::runtime_error("timestep cannot advance simulation time");
            const StepResult step = takeStep(stepSize);
            const double acceptedStep = step.acceptedTimeStep;
            dt = step.suggestedTimeStep;

            if (!std::isfinite(acceptedStep) || !(time + acceptedStep > time))
                throw std::runtime_error("accepted timestep cannot advance simulation time");
            time += acceptedStep;
            if (acceptedSteps == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("accepted-step counter overflow");
            ++acceptedSteps;

            const double roundingSlack =
                16.0 * std::numeric_limits<double>::epsilon() * std::abs(time);
            const bool finalFrame = time + roundingSlack >= params.endTime;
            if (finalFrame)
                time = params.endTime; // Avoid two final frames separated only by roundoff.
            const bool dipoleRemovalDue =
                params.dipoleRemoval &&
                (params.dipoleRemovalInterval == 0.0 ||
                 time + roundingSlack >= nextDipoleRemoval);
            if (deviceStepping && dipoleRemovalDue)
                synchronizeDeviceState();
            if (dipoleRemovalDue && dipoles.process(vortices) != 0) {
                integrator.invalidateCachedDerivative();
                if (deviceStepping) {
                    kernel->uploadDeviceState(vortices);
                    kernel->invalidateDeviceDerivative();
                    hostStateCurrent = true;
                }
                segmentReference = computeInvariants(vortices, *kernel);
            }
            if (dipoleRemovalDue && params.dipoleRemovalInterval > 0.0)
                advanceClock(nextDipoleRemoval, params.dipoleRemovalInterval);

            const bool trajectoryDue = time + roundingSlack >= nextOutput;
            const bool diagnosticsDue = time + roundingSlack >= schedule.nextDiagnosticsTime;
            const bool checkpointDue = time + roundingSlack >= schedule.nextCheckpointTime;
            // Advance every due clock before saving it, including coincident events.
            if (trajectoryDue)
                advanceClock(nextOutput, schedule.trajectoryInterval);
            if (diagnosticsDue)
                advanceClock(schedule.nextDiagnosticsTime, schedule.diagnosticsInterval);
            if (checkpointDue)
                advanceClock(schedule.nextCheckpointTime, schedule.checkpointInterval);
            if (trajectoryDue || diagnosticsDue || checkpointDue || finalFrame) {
                if (eventIndex == std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("output event-frame counter overflow");
                ++eventIndex;
            }
            if (trajectoryDue || finalFrame)
                writeTrajectory();
            if (diagnosticsDue || finalFrame)
                writeDiagnostics();
            if (checkpointDue || finalFrame) {
                if (outputIndex == std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("checkpoint index overflow");
                ++outputIndex;
                writeCurrentCheckpoint();
            }
        }

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
