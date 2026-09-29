#include "simulation.h"
#include "backend.h"
#include "checkpoint.h"
#include "compute.h"
#include "initial_condition.h"
#include "print.h"
#include "read.h"
#include "timestep.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

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
    const GeometrySignature geometry = geometrySignature(params);
    if (checkpoint.coreRadius != params.coreRadius || checkpoint.integrator != params.integrator ||
        checkpoint.boundary != params.boundary || checkpoint.geometryLengthX != geometry.lengthX ||
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
         (!params.dipoleRemovalUpper ||
          checkpoint.dipoleRemovalUpperDistance == params.dipoleRemovalUpperDistance));
    const bool scheduleMatches = !checkpoint.hasDipoleSchedule ||
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

    if (isPeriodicX(params.boundary) && metadata.boxLength) {
        const bool xMismatch = significantlyDifferent(*metadata.boxLength, params.boxLengthX);
        const bool yMismatch = isPeriodicY(params.boundary) &&
                               significantlyDifferent(*metadata.boxLength, params.boxLengthY);
        if (xMismatch || yMismatch) {
            throw std::invalid_argument(
                "initial-condition periodic length does not match simulation parameters");
        }
    }

    if (isDisk(params.boundary) && metadata.diskRadius &&
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

double nextScheduledTime(bool restarting, double currentTime, double interval, double savedInterval,
                         double savedNextTime) {
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
        params.outputTime, params.diagnosticsInterval(), params.checkpointInterval(),
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
        nextDipoleRemovalTime = preserveDipoleSchedule ? restart.nextDipoleRemovalTime
                                                       : currentTime + params.dipoleRemovalInterval;
        if (!std::isfinite(nextDipoleRemovalTime) || !(nextDipoleRemovalTime > currentTime)) {
            throw std::runtime_error("dipole-removal interval cannot advance simulation time");
        }
    }

    return {nextTrajectoryTime, output, nextDipoleRemovalTime};
}

void protectInputFiles(const std::filesystem::path &destination, const std::string &parameterFile,
                       const SimParams &params) {
    for (const std::string &input :
         {parameterFile, params.initialConditionFile, params.restartFile}) {
        if (!input.empty() && sameFile(destination, input))
            throw std::runtime_error("output path would overwrite input file: " + input);
    }
}

void validateCheckpointDestination(const RunPaths &paths, std::size_t index,
                                   const std::string &parameterFile, const SimParams &params) {
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
            throw std::runtime_error(
                "run directory already contains solver output: " + paths.directory.string() +
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

Checkpoint loadRestart(const SimParams &params) {
    if (params.restartFile.empty()) {
        Checkpoint checkpoint;
        return checkpoint;
    }
    Checkpoint checkpoint = loadCheckpoint(params.restartFile);
    if (!checkpointMatches(checkpoint, params))
        throw std::runtime_error(
            "checkpoint geometry or integrator settings do not match parameters");
    return checkpoint;
}

class SimulationRunner {
  public:
    SimulationRunner(SimParams params, std::string parameterFile)
        : params_(std::move(params)), parameterFile_(std::move(parameterFile)),
          paths_(runPaths(params_)), restarting_(!params_.restartFile.empty()),
          restart_(loadRestart(params_)),
          vortices_(restarting_ ? std::move(restart_.vortices) : makeInitialState(params_)),
          kernel_(makeBackendKernel(params_)),
          deviceStepper_(dynamic_cast<DeviceStepper *>(kernel_.get())),
          initial_(restarting_ ? restart_.initialInvariants
                               : computeInvariants(vortices_, *kernel_)),
          dipoles_(restarting_ ? DipoleManager(params_, restart_.dipoleState)
                               : DipoleManager(params_)),
          integrator_(0) {
        if (!restarting_)
            dipoles_.process(vortices_);
        segmentReference_ = restarting_ && restart_.hasSegmentInvariants
                                ? restart_.segmentInvariants
                                : computeInvariants(vortices_, *kernel_);
        integrator_ = RungeKuttaIntegrator(vortices_.size());
        velocity_.resize(vortices_.size());

        if (deviceStepper_)
            deviceStepper_->uploadState(vortices_);

        time_ = restarting_ ? restart_.time : 0.0;
        if (time_ > params_.endTime)
            throw std::runtime_error("checkpoint time is later than endTime");
        timeStep_ = startingTimeStep(params_, restart_, restarting_);
        const ScheduleState initialSchedule = makeSchedule(params_, restart_, restarting_, time_);
        nextTrajectoryTime_ = initialSchedule.nextTrajectoryTime;
        outputSchedule_ = initialSchedule.output;
        nextDipoleRemovalTime_ = initialSchedule.nextDipoleRemovalTime;
        acceptedSteps_ = restarting_ ? restart_.acceptedSteps : 0;
        checkpointIndex_ = restarting_ ? restart_.outputIndex : 0;
        eventIndex_ = restarting_ ? restart_.eventIndex : 0;

        if (restarting_ && restart_.outputIndex == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("checkpoint index overflow");
        const std::size_t firstCheckpointIndex = restarting_ ? restart_.outputIndex + 1 : 0;
        if (backendIsRoot()) {
            RunWriters writers = openRunWriters(paths_, params_, parameterFile_, initial_,
                                                outputSchedule_, firstCheckpointIndex);
            trajectory_ = std::move(writers.trajectory);
            diagnostics_ = std::move(writers.diagnostics);
        }
    }

    void run() {
        writeInitialOutputs();
        while (time_ < params_.endTime)
            advanceOneStep();
    }

  private:
    void synchronizeHostState() {
        if (deviceStepper_ && !hostStateCurrent_) {
            deviceStepper_->downloadState(vortices_);
            hostStateCurrent_ = true;
        }
    }

    void writeTrajectory() {
        synchronizeHostState();
        if (deviceStepper_)
            deviceStepper_->evaluateState(velocity_);
        else
            kernel_->evaluate(vortices_, velocity_);
        if (backendIsRoot())
            trajectory_->write(time_, eventIndex_, vortices_, velocity_);
    }

    void writeDiagnostics() {
        synchronizeHostState();
        const Invariants current = computeInvariants(vortices_, *kernel_);
        const DipoleEventState eventState = dipoles_.state();
        if (backendIsRoot()) {
            diagnostics_->write(time_, eventIndex_, current, segmentReference_,
                                eventState.removedPairs, eventState.removedUpperPairs,
                                eventState.reinjectedPairs);
            printDiagnostics(time_, acceptedSteps_, current, initial_, params_.boundary,
                             segmentReference_, eventState.removedPairs,
                             eventState.removedUpperPairs, eventState.reinjectedPairs);
        }
    }

    void writeCheckpoint() {
        synchronizeHostState();
        if (!backendIsRoot())
            return;
        validateCheckpointDestination(paths_, checkpointIndex_, parameterFile_, params_);
        ::writeCheckpoint(paths_.checkpoints, vortices_, initial_, params_, segmentReference_,
                          dipoles_.state(), outputSchedule_,
                          {time_, timeStep_, nextTrajectoryTime_, acceptedSteps_, checkpointIndex_,
                           eventIndex_, nextDipoleRemovalTime_});
    }

    void writeInitialOutputs() {
        writeTrajectory();
        writeDiagnostics();
        if (!restarting_)
            writeCheckpoint();
        if (backendIsRoot())
            writeRunProvenance(params_, parameterFile_, backendName(), backendRuntimeDetails(),
                               time_, eventIndex_, restarting_);
    }

    [[nodiscard]] double nextStepSize() const {
        const double timeToDipoleRemoval =
            params_.dipoleRemoval && params_.dipoleRemovalInterval > 0.0
                ? nextDipoleRemovalTime_ - time_
                : std::numeric_limits<double>::infinity();
        return std::min({timeStep_, params_.endTime - time_, nextTrajectoryTime_ - time_,
                         outputSchedule_.nextDiagnosticsTime - time_,
                         outputSchedule_.nextCheckpointTime - time_, timeToDipoleRemoval});
    }

    StepResult takeStep(double stepSize) {
        StepResult step{stepSize, timeStep_, 0.0, 0};
        if (deviceStepper_) {
            if (params_.integrator == IntegratorKind::rk4)
                deviceStepper_->rk4Step(stepSize);
            else
                step = deviceStepper_->dopri5Step(stepSize, params_.absoluteTolerance,
                                                  params_.relativeTolerance,
                                                  params_.minimumTimeStep, params_.maximumTimeStep);
            hostStateCurrent_ = false;
        } else if (params_.integrator == IntegratorKind::rk4) {
            integrator_.rk4Step(vortices_, stepSize, *kernel_);
        } else {
            step = integrator_.dopri5Step(vortices_, stepSize, *kernel_, params_);
        }
        return step;
    }

    void advanceClock(double &next, double interval) const {
        do {
            const double following = next + interval;
            if (!std::isfinite(following) || !(following > next))
                throw std::runtime_error("scheduled interval cannot advance simulation time");
            next = following;
        } while (next <= time_);
    }

    void processDipoleEvent(double roundingSlack) {
        const bool removalDue =
            params_.dipoleRemoval && (params_.dipoleRemovalInterval == 0.0 ||
                                      time_ + roundingSlack >= nextDipoleRemovalTime_);
        if (deviceStepper_ && removalDue)
            synchronizeHostState();
        if (removalDue && dipoles_.process(vortices_) != 0) {
            integrator_.invalidateCachedDerivative();
            if (deviceStepper_) {
                deviceStepper_->uploadState(vortices_);
                deviceStepper_->invalidateDerivative();
                hostStateCurrent_ = true;
            }
            segmentReference_ = computeInvariants(vortices_, *kernel_);
        }
        if (removalDue && params_.dipoleRemovalInterval > 0.0)
            advanceClock(nextDipoleRemovalTime_, params_.dipoleRemovalInterval);
    }

    void processOutputEvents(double roundingSlack, bool finalFrame) {
        const bool trajectoryDue = time_ + roundingSlack >= nextTrajectoryTime_;
        const bool diagnosticsDue = time_ + roundingSlack >= outputSchedule_.nextDiagnosticsTime;
        const bool checkpointDue = time_ + roundingSlack >= outputSchedule_.nextCheckpointTime;

        if (trajectoryDue)
            advanceClock(nextTrajectoryTime_, outputSchedule_.trajectoryInterval);
        if (diagnosticsDue)
            advanceClock(outputSchedule_.nextDiagnosticsTime, outputSchedule_.diagnosticsInterval);
        if (checkpointDue)
            advanceClock(outputSchedule_.nextCheckpointTime, outputSchedule_.checkpointInterval);

        if (trajectoryDue || diagnosticsDue || checkpointDue || finalFrame) {
            if (eventIndex_ == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("output event-frame counter overflow");
            ++eventIndex_;
        }
        if (trajectoryDue || finalFrame)
            writeTrajectory();
        if (diagnosticsDue || finalFrame)
            writeDiagnostics();
        if (checkpointDue || finalFrame) {
            if (checkpointIndex_ == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("checkpoint index overflow");
            ++checkpointIndex_;
            writeCheckpoint();
        }
    }

    void advanceOneStep() {
        const double requestedStep = nextStepSize();
        if (!(time_ + requestedStep > time_))
            throw std::runtime_error("timestep cannot advance simulation time");
        const StepResult step = takeStep(requestedStep);
        timeStep_ = step.suggestedTimeStep;
        if (!std::isfinite(step.acceptedTimeStep) || !(time_ + step.acceptedTimeStep > time_)) {
            throw std::runtime_error("accepted timestep cannot advance simulation time");
        }
        time_ += step.acceptedTimeStep;
        if (acceptedSteps_ == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("accepted-step counter overflow");
        ++acceptedSteps_;

        const double roundingSlack =
            16.0 * std::numeric_limits<double>::epsilon() * std::abs(time_);
        const bool finalFrame = time_ + roundingSlack >= params_.endTime;
        if (finalFrame)
            time_ = params_.endTime;
        processDipoleEvent(roundingSlack);
        processOutputEvents(roundingSlack, finalFrame);
    }

    SimParams params_;
    std::string parameterFile_;
    RunPaths paths_;
    bool restarting_;
    Checkpoint restart_;
    VortexSystem vortices_;
    std::unique_ptr<VelocityKernel> kernel_;
    DeviceStepper *deviceStepper_;
    Invariants initial_;
    DipoleManager dipoles_;
    Invariants segmentReference_;
    RungeKuttaIntegrator integrator_;
    VelocityField velocity_;
    bool hostStateCurrent_ = true;
    double time_ = 0.0;
    double timeStep_ = 0.0;
    double nextTrajectoryTime_ = 0.0;
    OutputSchedule outputSchedule_;
    double nextDipoleRemovalTime_ = 0.0;
    std::size_t acceptedSteps_ = 0;
    std::size_t checkpointIndex_ = 0;
    std::size_t eventIndex_ = 0;
    std::unique_ptr<TrajectoryWriter> trajectory_;
    std::unique_ptr<DiagnosticsWriter> diagnostics_;
};

} // namespace

void runSimulation(SimParams params, const std::string &parameterFile) {
    SimulationRunner(std::move(params), parameterFile).run();
}
