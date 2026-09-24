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
constexpr unsigned checkpointVersion = 6;
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
        output << "accepted_steps " << progress.acceptedSteps << '\n';
        output << "output_index " << progress.outputIndex << '\n';
        output << "event_index " << progress.eventIndex << '\n';
        output << "core_radius " << params.coreRadius << '\n';
        output << "integrator " << toString(params.integrator) << '\n';
        const bool periodicX =
            params.boundaryCondition == "periodic" || params.boundaryCondition == "periodic_x";
        const double geometryLengthX =
            periodicX ? params.boxLengthX
                      : (params.boundaryCondition == "disk" ? params.diskRadius : 0.0);
        const double geometryLengthY =
            params.boundaryCondition == "periodic" ? params.boxLengthY : 0.0;
        const int imageLayers =
            params.boundaryCondition == "periodic" ? params.periodicImageLayers : 0;
        output << "geometry " << params.boundaryCondition << ' ' << geometryLengthX << ' '
               << geometryLengthY << ' ' << imageLayers << '\n';
        output << "dipole_config " << params.dipoleRemoval << ' '
               << params.dipoleRemovalDistance << ' ' << toString(params.dipoleReinjection) << '\n';
        output << "dipole_counts " << dipoleState.removedPairs << ' ' << dipoleState.reinjectedPairs
               << '\n';
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
    Checkpoint c;
    std::string key, integratorName;
    std::size_t count = 0;
    auto require = [&](const char *expected) {
        if (!(input >> key) || key != expected)
            throw std::runtime_error(std::string("checkpoint is missing field: ") + expected);
    };
    const auto readSize = [&]() {
        std::string token;
        std::size_t value = 0;
        if (!(input >> token))
            throw std::runtime_error("checkpoint has a missing integer");
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
            throw std::runtime_error("checkpoint has an invalid non-negative integer");
        return value;
    };
    require("time");
    input >> c.time;
    require("suggested_time_step");
    input >> c.suggestedTimeStep;
    require("next_output_time");
    input >> c.nextOutputTime;
    if (fileVersion >= 5) {
        require("output_schedule");
        auto &schedule = c.outputSchedule;
        input >> schedule.trajectoryInterval >> schedule.diagnosticsInterval >>
            schedule.checkpointInterval >> schedule.nextDiagnosticsTime >>
            schedule.nextCheckpointTime;
        c.hasOutputSchedule = true;
    }
    require("accepted_steps");
    c.acceptedSteps = readSize();
    require("output_index");
    c.outputIndex = readSize();
    if (fileVersion >= 6) {
        require("event_index");
        c.eventIndex = readSize();
    } else {
        // Earlier checkpoints did not identify every independent output event.
        c.eventIndex = c.outputIndex;
    }
    require("core_radius");
    input >> c.coreRadius;
    require("integrator");
    input >> integratorName;
    const auto integrator = integratorFromString(integratorName);
    if (!integrator)
        throw std::runtime_error("unsupported checkpoint integrator: " + integratorName);
    c.integrator = *integrator;
    if (fileVersion >= 2) {
        require("geometry");
        input >> c.boundaryCondition >> c.geometryLengthX >> c.geometryLengthY >>
            c.periodicImageLayers;
    }
    if (fileVersion >= 3) {
        std::string reinjection;
        require("dipole_config");
        input >> c.dipoleRemoval >> c.dipoleRemovalDistance >> reinjection;
        const auto mode = reinjectionFromString(reinjection);
        if (!mode)
            throw std::runtime_error("unsupported checkpoint reinjection mode: " + reinjection);
        c.dipoleReinjection = *mode;
        require("dipole_counts");
        c.dipoleState.removedPairs = readSize();
        c.dipoleState.reinjectedPairs = readSize();
        require("random_engine_state");
        std::getline(input >> std::ws, c.dipoleState.randomEngineState);
    }
    require("initial_invariants");
    input >> c.initialInvariants.circulation >> c.initialInvariants.linearImpulseX >>
        c.initialInvariants.linearImpulseY >> c.initialInvariants.angularImpulse >>
        c.initialInvariants.hamiltonian;
    if (fileVersion >= 4) {
        require("segment_invariants");
        input >> c.segmentInvariants.circulation >> c.segmentInvariants.linearImpulseX >>
            c.segmentInvariants.linearImpulseY >> c.segmentInvariants.angularImpulse >>
            c.segmentInvariants.hamiltonian;
        c.hasSegmentInvariants = true;
    }
    require("vortex_count");
    count = readSize();
    require("vortices");
    // Do not allocate a corrupt declared population before checking that rows exist.
    for (std::size_t i = 0; i < count; ++i) {
        double x, y, gamma;
        if (!(input >> x >> y >> gamma))
            throw std::runtime_error("checkpoint vortex data is truncated or invalid");
        c.vortices.pushBack(x, y, gamma);
    }
    std::string trailing;
    if (input >> trailing)
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    if (input.bad())
        throw std::runtime_error("failed while reading checkpoint");
    input.clear();
    if (!input || !std::isfinite(c.time) || c.time < 0.0 || !std::isfinite(c.suggestedTimeStep) ||
        !(c.suggestedTimeStep > 0.0) || !std::isfinite(c.nextOutputTime) ||
        !(c.nextOutputTime > c.time) || !std::isfinite(c.coreRadius) || c.coreRadius < 0.0)
        throw std::runtime_error("checkpoint is truncated or invalid");
    if (c.hasOutputSchedule) {
        const auto &schedule = c.outputSchedule;
        for (double interval : {schedule.trajectoryInterval, schedule.diagnosticsInterval,
                                schedule.checkpointInterval})
            if (!std::isfinite(interval) || !(interval > 0.0))
                throw std::runtime_error("checkpoint has invalid output intervals");
        for (double next : {schedule.nextDiagnosticsTime, schedule.nextCheckpointTime})
            if (!std::isfinite(next) || !(next > c.time))
                throw std::runtime_error("checkpoint has invalid output schedule");
    }
    if (fileVersion >= 3 &&
        ((!std::isfinite(c.dipoleRemovalDistance) || c.dipoleRemovalDistance <= 0.0) ||
         c.dipoleState.randomEngineState.empty()))
        throw std::runtime_error("checkpoint has invalid dipole-removal state");
    for (const auto &invariants : {c.initialInvariants, c.segmentInvariants})
        for (double value :
             {invariants.circulation, invariants.linearImpulseX, invariants.linearImpulseY,
              invariants.angularImpulse, invariants.hamiltonian})
            if (!std::isfinite(value))
                throw std::runtime_error("checkpoint contains non-finite invariants");
    if (c.dipoleState.reinjectedPairs > c.dipoleState.removedPairs)
        throw std::runtime_error("checkpoint has invalid dipole event counts");
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(c.vortices.x[i]) || !std::isfinite(c.vortices.y[i]) ||
            !std::isfinite(c.vortices.circulation[i]))
            throw std::runtime_error("checkpoint contains non-finite vortex data");
    return c;
}
