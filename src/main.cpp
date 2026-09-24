#include "backend.h"
#include "checkpoint.h"
#include "compute.h"
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

bool checkpointMatches(const Checkpoint &checkpoint, const SimParams &params) {
    const bool periodicX =
        params.boundaryCondition == "periodic" || params.boundaryCondition == "periodic_x";
    const double lengthX = periodicX
                               ? params.boxLengthX
                               : (params.boundaryCondition == "disk" ? params.diskRadius : 0.0);
    const double lengthY = params.boundaryCondition == "periodic" ? params.boxLengthY : 0.0;
    const int imageLayers = params.boundaryCondition == "periodic" ? params.periodicImageLayers : 0;
    return checkpoint.coreRadius == params.coreRadius &&
           checkpoint.integrator == params.integrator &&
           checkpoint.boundaryCondition == params.boundaryCondition &&
           checkpoint.geometryLengthX == lengthX && checkpoint.geometryLengthY == lengthY &&
           checkpoint.periodicImageLayers == imageLayers &&
           checkpoint.dipoleRemoval == params.dipoleRemoval &&
           (!params.dipoleRemoval ||
            (checkpoint.dipoleRemovalDistance == params.dipoleRemovalDistance &&
             checkpoint.dipoleReinjection == params.dipoleReinjection));
}

VortexSystem makeInitialState(const SimParams &params) {
    if (!params.initialConditionFile.empty()) {
        const InitialConditionMetadata metadata =
            readInitialConditionMetadata(params.initialConditionFile);
        if (metadata.geometry && *metadata.geometry != params.boundaryCondition)
            throw std::invalid_argument("initial-condition geometry is " + *metadata.geometry +
                                        " but boundaryCondition is " + params.boundaryCondition);
        if ((params.boundaryCondition == "periodic" || params.boundaryCondition == "periodic_x") &&
            metadata.boxLength) {
            const bool xMismatch = std::abs(*metadata.boxLength - params.boxLengthX) >
                                   1e-13 * std::max(*metadata.boxLength, params.boxLengthX);
            const bool yMismatch = params.boundaryCondition == "periodic" &&
                                   std::abs(*metadata.boxLength - params.boxLengthY) >
                                       1e-13 * std::max(*metadata.boxLength, params.boxLengthY);
            if (xMismatch || yMismatch)
                throw std::invalid_argument(
                    "initial-condition periodic length does not match simulation parameters");
        }
        if (params.boundaryCondition == "disk" && metadata.diskRadius &&
            std::abs(*metadata.diskRadius - params.diskRadius) >
                1e-13 * std::max(*metadata.diskRadius, params.diskRadius))
            throw std::invalid_argument(
                "initial-condition disk radius does not match simulation parameters");
        return loadVortices(params.initialConditionFile);
    }

    VortexSystem vortices(params.vortexCount);
    if (params.boundaryCondition == "periodic") {
        initializePeriodicVortices(vortices, params.boxLengthX, params.boxLengthY,
                                   params.randomSeed);
        return vortices;
    }

    // Keep the demonstration ring away from the disk wall.
    double radius = 1.0;
    if (params.boundaryCondition == "disk")
        radius = 0.5 * params.diskRadius;
    initializeVortices(vortices, radius);
    return vortices;
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
        double dt =
            restarting
                ? restart.suggestedTimeStep
                : (params.integrator == IntegratorKind::dopri5
                       ? std::clamp(params.timeStep, params.minimumTimeStep, params.maximumTimeStep)
                       : params.timeStep);
        // Preserve saved schedules when the interval is unchanged. A changed interval
        // starts a new cadence from the restart time. Legacy checkpoints have one clock.
        const auto nextTime = [&](double interval, double savedInterval, double savedNext) {
            const double next =
                restarting && interval == savedInterval ? savedNext : time + interval;
            if (!std::isfinite(next) || !(next > time))
                throw std::runtime_error("output interval cannot advance simulation time");
            return next;
        };
        const auto &saved = restart.outputSchedule;
        const bool savedSchedule = restarting && restart.hasOutputSchedule;
        double nextOutput = nextTime(params.outputTime,
                                     savedSchedule ? saved.trajectoryInterval : params.outputTime,
                                     restart.nextOutputTime);
        OutputSchedule schedule{
            params.outputTime, params.diagnosticsInterval(), params.checkpointInterval(),
            nextTime(params.diagnosticsInterval(),
                     savedSchedule ? saved.diagnosticsInterval : params.outputTime,
                     savedSchedule ? saved.nextDiagnosticsTime : restart.nextOutputTime),
            nextTime(params.checkpointInterval(),
                     savedSchedule ? saved.checkpointInterval : params.outputTime,
                     savedSchedule ? saved.nextCheckpointTime : restart.nextOutputTime)};
        std::size_t acceptedSteps = restarting ? restart.acceptedSteps : 0;
        // The filename index counts checkpoints, independently of CSV frames.
        std::size_t outputIndex = restarting ? restart.outputIndex : 0;
        // One event frame identifies a simultaneous trajectory/diagnostics/checkpoint save.
        std::size_t eventIndex = restarting ? restart.eventIndex : 0;

        const auto protectInput = [&](const std::filesystem::path &destination) {
            for (const auto &input :
                 {parameterFile, params.initialConditionFile, params.restartFile})
                if (!input.empty() && sameFile(destination, input))
                    throw std::runtime_error("output path would overwrite input file: " + input);
        };
        const auto checkCheckpointDestination = [&](std::size_t index) {
            const auto destination = checkpointPath(paths.checkpoints.string(), index);
            protectInput(destination);
            if (sameFile(destination, paths.trajectory) || sameFile(destination, paths.diagnostics))
                throw std::runtime_error("checkpoint and CSV output paths must be different");
        };
        // Detect the common rerun/restart collision before opening and possibly truncating CSVs.
        if (restarting && restart.outputIndex == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("checkpoint index overflow");
        const std::size_t firstCheckpointIndex = restarting ? restart.outputIndex + 1 : 0;
        std::unique_ptr<TrajectoryWriter> trajectory;
        std::unique_ptr<DiagnosticsWriter> diagnostics;
        if (backendIsRoot()) {
            // Check all managed destinations before replacing any artefact.
            const auto trajectoryPath = std::filesystem::weakly_canonical(paths.trajectory);
            const auto diagnosticsPath = std::filesystem::weakly_canonical(paths.diagnostics);
            if (sameFile(trajectoryPath, diagnosticsPath))
                throw std::runtime_error(
                    "managed trajectory and diagnostics paths must be different");
            protectInput(trajectoryPath);
            protectInput(diagnosticsPath);
            checkCheckpointDestination(firstCheckpointIndex);
            if (hasManagedRunOutput(paths)) {
                if (!params.overwriteRun)
                    throw std::runtime_error(
                        "run directory already contains solver output: " +
                        paths.directory.string() +
                        " (choose another runDirectory or set overwriteRun true)");
                removeManagedRunOutput(paths);
            }
            trajectory = std::make_unique<TrajectoryWriter>(paths.trajectory.string(), false);
            diagnostics =
                std::make_unique<DiagnosticsWriter>(paths.diagnostics.string(), initial, false);
            std::cout << "backend=" << backendName() << '\n'
                      << "trajectory="
                      << std::filesystem::absolute(paths.trajectory).lexically_normal()
                      << " interval=" << schedule.trajectoryInterval << '\n'
                      << "diagnostics="
                      << std::filesystem::absolute(paths.diagnostics).lexically_normal()
                      << " interval=" << schedule.diagnosticsInterval << '\n'
                      << "checkpoints="
                      << std::filesystem::absolute(paths.checkpoints).lexically_normal()
                      << " interval=" << schedule.checkpointInterval << '\n'
                      << std::flush;
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
                                   eventState.removedPairs, eventState.reinjectedPairs);
                printDiagnostics(time, acceptedSteps, current, initial, params.boundaryCondition,
                                 segmentReference, eventState.removedPairs,
                                 eventState.reinjectedPairs);
            }
        };
        const auto writeCurrentCheckpoint = [&] {
            synchronizeDeviceState();
            if (backendIsRoot()) {
                checkCheckpointDestination(outputIndex);
                writeCheckpoint(paths.checkpoints, vortices, initial, params, segmentReference,
                                dipoles.state(), schedule,
                                {time, dt, nextOutput, acceptedSteps, outputIndex, eventIndex});
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
            StepResult result{stepSize, dt, 0.0, 0};
            if (deviceStepping) {
                if (params.integrator == IntegratorKind::rk4)
                    kernel->deviceRk4Step(stepSize);
                else
                    result = kernel->deviceDopri5Step(
                        stepSize, params.absoluteTolerance, params.relativeTolerance,
                        params.minimumTimeStep, params.maximumTimeStep);
                hostStateCurrent = false;
            } else if (params.integrator == IntegratorKind::rk4) {
                integrator.rk4Step(vortices, stepSize, *kernel);
            } else {
                result = integrator.dopri5Step(vortices, stepSize, *kernel, params);
            }
            return result;
        };

        while (time < params.endTime) {
            // Land on the next event from any output stream, or the final time.
            const double stepSize =
                std::min({dt, params.endTime - time, nextOutput - time,
                          schedule.nextDiagnosticsTime - time, schedule.nextCheckpointTime - time});
            if (!(time + stepSize > time))
                throw std::runtime_error("timestep cannot advance simulation time");
            const StepResult result = takeStep(stepSize);
            const double acceptedStep = result.acceptedTimeStep;
            dt = result.suggestedTimeStep;

            if (!std::isfinite(acceptedStep) || !(time + acceptedStep > time))
                throw std::runtime_error("accepted timestep cannot advance simulation time");
            time += acceptedStep;
            if (acceptedSteps == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("accepted-step counter overflow");
            ++acceptedSteps;
            if (deviceStepping && params.dipoleRemoval)
                synchronizeDeviceState();
            if (dipoles.process(vortices) != 0) {
                integrator.invalidateCachedDerivative();
                if (deviceStepping) {
                    kernel->uploadDeviceState(vortices);
                    kernel->invalidateDeviceDerivative();
                    hostStateCurrent = true;
                }
                segmentReference = computeInvariants(vortices, *kernel);
            }

            const double roundingSlack =
                16.0 * std::numeric_limits<double>::epsilon() * std::abs(time);
            const bool finalFrame = time + roundingSlack >= params.endTime;
            if (finalFrame)
                time = params.endTime; // Avoid two final frames separated only by roundoff.
            const bool trajectoryDue = time + roundingSlack >= nextOutput;
            const bool diagnosticsDue = time + roundingSlack >= schedule.nextDiagnosticsTime;
            const bool checkpointDue = time + roundingSlack >= schedule.nextCheckpointTime;
            const auto advance = [&](double &next, double interval) {
                do {
                    const double following = next + interval;
                    if (!std::isfinite(following) || !(following > next))
                        throw std::runtime_error("output interval cannot advance simulation time");
                    next = following;
                } while (next <= time);
            };
            // Advance every due clock before saving it, including coincident events.
            if (trajectoryDue)
                advance(nextOutput, schedule.trajectoryInterval);
            if (diagnosticsDue)
                advance(schedule.nextDiagnosticsTime, schedule.diagnosticsInterval);
            if (checkpointDue)
                advance(schedule.nextCheckpointTime, schedule.checkpointInterval);
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
