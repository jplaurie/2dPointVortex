#include "checkpoint.h"
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace {
constexpr const char *checkpointMagic = "POINT_VORTEX_CHECKPOINT";
constexpr unsigned checkpointVersion = 8;

void validateCheckpointContents(const Checkpoint &checkpoint, unsigned fileVersion) {
    if (!std::isfinite(checkpoint.time) || checkpoint.time < 0.0 ||
        !std::isfinite(checkpoint.suggestedTimeStep) ||
        !(checkpoint.suggestedTimeStep > 0.0) ||
        !std::isfinite(checkpoint.nextOutputTime) ||
        !(checkpoint.nextOutputTime > checkpoint.time) ||
        !std::isfinite(checkpoint.coreRadius) || checkpoint.coreRadius < 0.0) {
        throw std::runtime_error("checkpoint is truncated or invalid");
    }

    if (checkpoint.hasOutputSchedule) {
        const OutputSchedule &schedule = checkpoint.outputSchedule;
        for (double interval : {schedule.trajectoryInterval, schedule.diagnosticsInterval,
                                schedule.checkpointInterval}) {
            if (!std::isfinite(interval) || !(interval > 0.0))
                throw std::runtime_error("checkpoint has invalid output intervals");
        }
        for (double nextTime : {schedule.nextDiagnosticsTime, schedule.nextCheckpointTime}) {
            if (!std::isfinite(nextTime) || !(nextTime > checkpoint.time))
                throw std::runtime_error("checkpoint has invalid output schedule");
        }
    }

    if (fileVersion >= 3 &&
        ((!std::isfinite(checkpoint.dipoleRemovalDistance) ||
          checkpoint.dipoleRemovalDistance <= 0.0) ||
         checkpoint.dipoleState.randomEngineState.empty())) {
        throw std::runtime_error("checkpoint has invalid dipole-removal state");
    }
    if (fileVersion >= 8 &&
        ((!std::isfinite(checkpoint.dipoleRemovalUpperDistance) ||
          checkpoint.dipoleRemovalUpperDistance <= 0.0) ||
         (checkpoint.dipoleRemovalUpper &&
          (!checkpoint.dipoleRemoval || !(checkpoint.dipoleRemovalUpperDistance >
                                           checkpoint.dipoleRemovalDistance))))) {
        throw std::runtime_error("checkpoint has invalid upper dipole-removal state");
    }
    if (checkpoint.hasDipoleSchedule &&
        (!std::isfinite(checkpoint.dipoleRemovalInterval) ||
         checkpoint.dipoleRemovalInterval < 0.0 ||
         (checkpoint.dipoleRemoval && checkpoint.dipoleRemovalInterval > 0.0 &&
          (!std::isfinite(checkpoint.nextDipoleRemovalTime) ||
           !(checkpoint.nextDipoleRemovalTime > checkpoint.time))))) {
        throw std::runtime_error("checkpoint has invalid dipole-removal schedule");
    }

    for (const Invariants &invariants :
         {checkpoint.initialInvariants, checkpoint.segmentInvariants}) {
        for (double value :
             {invariants.circulation, invariants.linearImpulseX, invariants.linearImpulseY,
              invariants.angularImpulse, invariants.hamiltonian}) {
            if (!std::isfinite(value))
                throw std::runtime_error("checkpoint contains non-finite invariants");
        }
    }

    if (checkpoint.dipoleState.reinjectedPairs > checkpoint.dipoleState.removedPairs)
        throw std::runtime_error("checkpoint has invalid dipole event counts");
    if (checkpoint.dipoleState.removedUpperPairs > checkpoint.dipoleState.removedPairs)
        throw std::runtime_error("checkpoint has invalid upper dipole event count");

    for (std::size_t index = 0; index < checkpoint.vortices.size(); ++index) {
        if (!std::isfinite(checkpoint.vortices.x[index]) ||
            !std::isfinite(checkpoint.vortices.y[index]) ||
            !std::isfinite(checkpoint.vortices.circulation[index])) {
            throw std::runtime_error("checkpoint contains non-finite vortex data");
        }
    }
}
} // namespace
std::filesystem::path checkpointPath(const std::filesystem::path &directory,
                                     std::size_t outputIndex) {
    std::ostringstream filename;
    filename << "checkpoint_" << std::setw(8) << std::setfill('0') << outputIndex << ".dat";
    return directory / filename.str();
}
void writeCheckpoint(const std::filesystem::path &directory, const VortexSystem &vortices,
                     const Invariants &initialInvariants, const SimParams &params,
                     const Invariants &segmentInvariants, const DipoleEventState &dipoleState,
                     const OutputSchedule &outputSchedule, const CheckpointProgress &progress,
                     bool overwrite) {
    vortices.validate();
    std::filesystem::create_directories(directory);
    const auto destination = checkpointPath(directory, progress.outputIndex);
    if (!overwrite && std::filesystem::exists(destination))
        throw std::runtime_error("refusing to overwrite checkpoint: " + destination.string());
    auto temporary = destination;
    temporary += ".tmp";
    {
        // The destination appears only after the complete temporary file is closed.
        std::ofstream output(temporary, std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot create checkpoint: " + temporary.string());
        output << std::setprecision(17);
        output << checkpointMagic << ' ' << checkpointVersion << '\n';
        output << "time " << progress.time << '\n';
        output << "suggested_time_step " << progress.suggestedTimeStep << '\n';
        output << "next_output_time " << progress.nextOutputTime << '\n';
        output << "output_schedule " << outputSchedule.trajectoryInterval << ' '
               << outputSchedule.diagnosticsInterval << ' ' << outputSchedule.checkpointInterval
               << ' ' << outputSchedule.nextDiagnosticsTime << ' '
               << outputSchedule.nextCheckpointTime << '\n';
        output << "dipole_schedule " << params.dipoleRemovalInterval << ' '
               << progress.nextDipoleRemovalTime << '\n';
        output << "accepted_steps " << progress.acceptedSteps << '\n';
        output << "output_index " << progress.outputIndex << '\n';
        output << "event_index " << progress.eventIndex << '\n';
        output << "core_radius " << params.coreRadius << '\n';
        output << "integrator " << toString(params.integrator) << '\n';
        const GeometrySignature geometry = geometrySignature(params);
        output << "geometry " << toString(params.boundary) << ' ' << geometry.lengthX << ' '
               << geometry.lengthY << ' ' << geometry.imageLayers << '\n';
        output << "dipole_config " << params.dipoleRemoval << ' '
               << params.dipoleRemovalDistance << ' ' << params.dipoleRemovalUpper << ' '
               << params.dipoleRemovalUpperDistance << ' '
               << toString(params.dipoleReinjection) << '\n';
        output << "dipole_counts " << dipoleState.removedPairs << ' '
               << dipoleState.removedUpperPairs << ' ' << dipoleState.reinjectedPairs << '\n';
        output << "random_engine_state " << dipoleState.randomEngineState << '\n';
        output << "initial_invariants " << initialInvariants.circulation << ' '
               << initialInvariants.linearImpulseX << ' ' << initialInvariants.linearImpulseY << ' '
               << initialInvariants.angularImpulse << ' ' << initialInvariants.hamiltonian << '\n';
        output << "segment_invariants " << segmentInvariants.circulation << ' '
               << segmentInvariants.linearImpulseX << ' ' << segmentInvariants.linearImpulseY << ' '
               << segmentInvariants.angularImpulse << ' ' << segmentInvariants.hamiltonian << '\n';
        output << "vortex_count " << vortices.size() << '\n';
        output << "vortices\n";
        for (std::size_t i = 0; i < vortices.size(); ++i)
            output << vortices.x[i] << ' ' << vortices.y[i] << ' ' << vortices.circulation[i]
                   << '\n';
        output.flush();
        if (!output)
            throw std::runtime_error("failed while writing checkpoint: " + temporary.string());
    }
#if defined(__unix__) || defined(__APPLE__)
    // fsync the file before rename and the directory after rename for power-loss durability.
    const int fileDescriptor = ::open(temporary.c_str(), O_RDONLY);
    if (fileDescriptor < 0 || ::fsync(fileDescriptor) != 0) {
        if (fileDescriptor >= 0)
            ::close(fileDescriptor);
        throw std::runtime_error("cannot synchronize checkpoint: " + temporary.string());
    }
    ::close(fileDescriptor);
#endif
    std::filesystem::rename(temporary, destination);
#if defined(__unix__) || defined(__APPLE__)
    const int directoryDescriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (directoryDescriptor >= 0) {
        ::fsync(directoryDescriptor);
        ::close(directoryDescriptor);
    }
#endif
}
Checkpoint loadCheckpoint(const std::filesystem::path &filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open checkpoint: " + filename.string());
    std::string marker;
    unsigned fileVersion = 0;
    if (!(input >> marker >> fileVersion) || marker != checkpointMagic ||
        (fileVersion < 1 || fileVersion > checkpointVersion))
        throw std::runtime_error("unsupported or corrupt checkpoint header");
    Checkpoint checkpoint;
    std::string key, integratorName;
    std::size_t vortexCount = 0;
    auto require = [&](const char *expected) {
        if (!(input >> key) || key != expected)
            throw std::runtime_error(std::string("checkpoint is missing field: ") + expected);
    };
    const auto readSize = [&]() {
        std::string token;
        std::size_t value = 0;
        if (!(input >> token))
            throw std::runtime_error("checkpoint has a missing integer");
        const auto conversion = std::from_chars(token.data(), token.data() + token.size(), value);
        if (conversion.ec != std::errc{} || conversion.ptr != token.data() + token.size())
            throw std::runtime_error("checkpoint has an invalid non-negative integer");
        return value;
    };
    require("time");
    input >> checkpoint.time;
    require("suggested_time_step");
    input >> checkpoint.suggestedTimeStep;
    require("next_output_time");
    input >> checkpoint.nextOutputTime;
    if (fileVersion >= 5) {
        require("output_schedule");
        OutputSchedule &schedule = checkpoint.outputSchedule;
        input >> schedule.trajectoryInterval >> schedule.diagnosticsInterval >>
            schedule.checkpointInterval >> schedule.nextDiagnosticsTime >>
            schedule.nextCheckpointTime;
        checkpoint.hasOutputSchedule = true;
    }
    if (fileVersion >= 7) {
        require("dipole_schedule");
        input >> checkpoint.dipoleRemovalInterval >> checkpoint.nextDipoleRemovalTime;
        checkpoint.hasDipoleSchedule = true;
    }
    require("accepted_steps");
    checkpoint.acceptedSteps = readSize();
    require("output_index");
    checkpoint.outputIndex = readSize();
    if (fileVersion >= 6) {
        require("event_index");
        checkpoint.eventIndex = readSize();
    } else {
        // Earlier checkpoints did not identify every independent output event.
        checkpoint.eventIndex = checkpoint.outputIndex;
    }
    require("core_radius");
    input >> checkpoint.coreRadius;
    require("integrator");
    input >> integratorName;
    const auto integrator = integratorFromString(integratorName);
    if (!integrator)
        throw std::runtime_error("unsupported checkpoint integrator: " + integratorName);
    checkpoint.integrator = *integrator;
    if (fileVersion >= 2) {
        std::string boundaryName;
        require("geometry");
        input >> boundaryName >> checkpoint.geometryLengthX >> checkpoint.geometryLengthY >>
            checkpoint.periodicImageLayers;
        const auto boundary = boundaryFromString(boundaryName);
        if (!boundary)
            throw std::runtime_error("unsupported checkpoint boundary: " + boundaryName);
        checkpoint.boundary = *boundary;
    }
    if (fileVersion >= 3) {
        std::string reinjection;
        require("dipole_config");
        input >> checkpoint.dipoleRemoval >> checkpoint.dipoleRemovalDistance;
        if (fileVersion >= 8) {
            input >> checkpoint.dipoleRemovalUpper >> checkpoint.dipoleRemovalUpperDistance;
            checkpoint.hasDipoleUpperConfig = true;
        }
        input >> reinjection;
        const auto mode = reinjectionFromString(reinjection);
        if (!mode)
            throw std::runtime_error("unsupported checkpoint reinjection mode: " + reinjection);
        checkpoint.dipoleReinjection = *mode;
        require("dipole_counts");
        checkpoint.dipoleState.removedPairs = readSize();
        if (fileVersion >= 8)
            checkpoint.dipoleState.removedUpperPairs = readSize();
        checkpoint.dipoleState.reinjectedPairs = readSize();
        require("random_engine_state");
        std::getline(input >> std::ws, checkpoint.dipoleState.randomEngineState);
    }
    require("initial_invariants");
    input >> checkpoint.initialInvariants.circulation >>
        checkpoint.initialInvariants.linearImpulseX >> checkpoint.initialInvariants.linearImpulseY >>
        checkpoint.initialInvariants.angularImpulse >> checkpoint.initialInvariants.hamiltonian;
    if (fileVersion >= 4) {
        require("segment_invariants");
        input >> checkpoint.segmentInvariants.circulation >>
            checkpoint.segmentInvariants.linearImpulseX >>
            checkpoint.segmentInvariants.linearImpulseY >>
            checkpoint.segmentInvariants.angularImpulse >>
            checkpoint.segmentInvariants.hamiltonian;
        checkpoint.hasSegmentInvariants = true;
    }
    require("vortex_count");
    vortexCount = readSize();
    require("vortices");
    // Do not allocate a corrupt declared population before checking that rows exist.
    for (std::size_t index = 0; index < vortexCount; ++index) {
        double x, y, gamma;
        if (!(input >> x >> y >> gamma))
            throw std::runtime_error("checkpoint vortex data is truncated or invalid");
        checkpoint.vortices.pushBack(x, y, gamma);
    }
    std::string trailing;
    if (input >> trailing)
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    if (input.bad())
        throw std::runtime_error("failed while reading checkpoint");
    input.clear();
    validateCheckpointContents(checkpoint, fileVersion);
    return checkpoint;
}
