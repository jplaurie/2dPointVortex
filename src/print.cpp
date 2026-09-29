#include "print.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::ofstream openOutput(const std::string &filename, bool overwrite, const char *description) {
    if (!overwrite && std::filesystem::exists(filename))
        throw std::runtime_error("refusing to overwrite " + std::string(description) + ": " +
                                 filename);
    const auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream output(filename, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot open " + std::string(description) + ": " + filename);
    output << std::setprecision(17);
    return output;
}
} // namespace

TrajectoryWriter::TrajectoryWriter(const std::string &filename, bool overwrite)
    : output_(openOutput(filename, overwrite, "output file")) {
    output_ << "time,frame,index,x,y,circulation,u,v\n";
    output_.flush();
}
void TrajectoryWriter::write(double time, std::size_t frame, const VortexSystem &vortices,
                             const VelocityField &velocity) {
    if (velocity.x.size() != vortices.size() || velocity.y.size() != vortices.size())
        throw std::invalid_argument("velocity and vortex arrays have different lengths");
    for (std::size_t i = 0; i < vortices.size(); ++i)
        output_ << time << ',' << frame << ',' << i << ',' << vortices.x[i] << ',' << vortices.y[i]
                << ',' << vortices.circulation[i] << ',' << velocity.x[i] << ',' << velocity.y[i]
                << '\n';
    output_.flush();
    if (!output_)
        throw std::runtime_error("failed while writing trajectory output");
}
DiagnosticsWriter::DiagnosticsWriter(const std::string &filename, const Invariants &initial,
                                     bool overwrite)
    : output_(openOutput(filename, overwrite, "diagnostics file")), initial_(initial) {
    output_
        << "time,frame,circulation,linear_impulse_x,linear_impulse_y,angular_impulse,hamiltonian,"
           "delta_circulation,delta_linear_impulse_x,delta_linear_impulse_y,"
           "delta_angular_impulse,delta_hamiltonian,segment_delta_circulation,"
           "segment_delta_linear_impulse_x,segment_delta_linear_impulse_y,"
           "segment_delta_angular_impulse,segment_delta_hamiltonian,removed_pairs,"
           "removed_upper_pairs,reinjected_pairs\n";
    output_.flush();
}
void DiagnosticsWriter::write(double time, std::size_t frame, const Invariants &current,
                              const Invariants &segmentReference, std::size_t removedPairs,
                              std::size_t removedUpperPairs, std::size_t reinjectedPairs) {
    output_ << time << ',' << frame << ',' << current.circulation << ',' << current.linearImpulseX
            << ',' << current.linearImpulseY << ',' << current.angularImpulse << ','
            << current.hamiltonian << ',' << current.circulation - initial_.circulation << ','
            << current.linearImpulseX - initial_.linearImpulseX << ','
            << current.linearImpulseY - initial_.linearImpulseY << ','
            << current.angularImpulse - initial_.angularImpulse << ','
            << current.hamiltonian - initial_.hamiltonian << ','
            << current.circulation - segmentReference.circulation << ','
            << current.linearImpulseX - segmentReference.linearImpulseX << ','
            << current.linearImpulseY - segmentReference.linearImpulseY << ','
            << current.angularImpulse - segmentReference.angularImpulse << ','
            << current.hamiltonian - segmentReference.hamiltonian << ',' << removedPairs << ','
            << removedUpperPairs << ',' << reinjectedPairs << '\n';
    output_.flush();
    if (!output_)
        throw std::runtime_error("failed while writing diagnostics output");
}
namespace {
void writeRecord(const std::filesystem::path &path, const SimParams &params,
                 const std::string &parameterFile, const std::string &backend,
                 const std::string &runtimeDetails, double startTime, std::size_t startFrame,
                 bool restarting, const std::filesystem::path &segment) {
    const auto absolutePath = [](const std::string &value) {
        return value.empty() ? std::string{}
                             : std::filesystem::absolute(value).lexically_normal().string();
    };
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write run record: " + temporary.string());
    const RunPaths paths = runPaths(params);
    output << std::setprecision(17) << "POINT_VORTEX_RUN_RECORD 1\n"
           << "parameter_file " << std::quoted(absolutePath(parameterFile)) << '\n'
           << "run_directory " << std::quoted(absolutePath(params.runDirectory)) << '\n'
           << "backend " << backend << '\n'
           << runtimeDetails << '\n'
           << "start_time " << startTime << '\n'
           << "start_frame " << startFrame << '\n'
           << "restarting " << restarting << '\n'
           << "restart_file " << std::quoted(absolutePath(params.restartFile)) << '\n'
           << "N " << params.vortexCount << '\n'
           << "timeStep " << params.timeStep << '\n'
           << "endTime " << params.endTime << '\n'
           << "outputTime " << params.outputTime << '\n'
           << "diagnosticsTime " << params.diagnosticsInterval() << '\n'
           << "checkpointTime " << params.checkpointInterval() << '\n'
           << "integrator " << toString(params.integrator) << '\n'
           << "coreRadius " << params.coreRadius << '\n'
           << "numThreads " << params.numThreads << '\n'
           << "boundaryCondition " << toString(params.boundary) << '\n'
           << "boxLengthX " << params.boxLengthX << '\n'
           << "boxLengthY " << params.boxLengthY << '\n'
           << "periodicImageLayers " << params.periodicImageLayers << '\n'
           << "diskRadius " << params.diskRadius << '\n'
           << "randomSeed " << params.randomSeed << '\n'
           << "dipoleRemoval " << params.dipoleRemoval << '\n'
           << "dipoleRemovalDistance " << params.dipoleRemovalDistance << '\n'
           << "dipoleRemovalUpper " << params.dipoleRemovalUpper << '\n'
           << "dipoleRemovalUpperDistance " << params.dipoleRemovalUpperDistance << '\n'
           << "dipoleRemovalInterval " << params.dipoleRemovalInterval << '\n'
           << "dipoleReinjection " << toString(params.dipoleReinjection) << '\n'
           << "initialCondition " << toString(params.initialCondition) << '\n'
           << "initialConditionFile " << std::quoted(absolutePath(params.initialConditionFile))
           << '\n'
           << "trajectory_file " << std::quoted(absolutePath(paths.trajectory.string())) << '\n'
           << "diagnostics_file " << std::quoted(absolutePath(paths.diagnostics.string())) << '\n'
           << "checkpoint_directory " << std::quoted(absolutePath(paths.checkpoints.string()))
           << '\n'
           << "segmentDirectory "
           << std::quoted(std::filesystem::absolute(segment).lexically_normal().string()) << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("failed while writing run record: " + temporary.string());
    std::filesystem::rename(temporary, path);
}
} // namespace
void writeRunProvenance(const SimParams &params, const std::string &parameterFile,
                        const std::string &backend, const std::string &runtimeDetails,
                        double startTime, std::size_t startFrame, bool restarting) {
    const std::filesystem::path root = runPaths(params).directory;
    const auto segments = root / "segments";
    std::filesystem::create_directories(segments);
    std::filesystem::path segment;
    for (std::size_t index = 1;; ++index) {
        std::ostringstream name;
        name << "segment_" << std::setw(8) << std::setfill('0') << index;
        segment = segments / name.str();
        if (std::filesystem::create_directory(segment))
            break;
    }
    writeRecord(segment / "resolved_parameters.txt", params, parameterFile, backend, runtimeDetails,
                startTime, startFrame, restarting, segment);
    writeRecord(root / "resolved_parameters.txt", params, parameterFile, backend, runtimeDetails,
                startTime, startFrame, restarting, segment);
}
void printDiagnostics(double time, std::size_t steps, const Invariants &current,
                      const Invariants &initial, BoundaryKind boundary,
                      const Invariants &segmentReference, std::size_t removedPairs,
                      std::size_t removedUpperPairs, std::size_t reinjectedPairs) {
    std::cout << std::setprecision(10) << "time=" << time << " steps=" << steps
              << " circulation=" << current.circulation
              << " dCirculation=" << current.circulation - initial.circulation
              << " segmentDCirculation=" << current.circulation - segmentReference.circulation
              << " H=" << current.hamiltonian
              << " dH=" << current.hamiltonian - initial.hamiltonian
              << " segmentDH=" << current.hamiltonian - segmentReference.hamiltonian;

    if (boundary == BoundaryKind::infinite || boundary == BoundaryKind::periodic_x ||
        boundary == BoundaryKind::periodic) {
        std::cout << " Ix=" << current.linearImpulseX
                  << " dIx=" << current.linearImpulseX - initial.linearImpulseX
                  << " segmentDIx=" << current.linearImpulseX - segmentReference.linearImpulseX
                  << " Iy=" << current.linearImpulseY
                  << " dIy=" << current.linearImpulseY - initial.linearImpulseY
                  << " segmentDIy=" << current.linearImpulseY - segmentReference.linearImpulseY;
    }
    if (boundary == BoundaryKind::infinite || boundary == BoundaryKind::disk) {
        std::cout << " L=" << current.angularImpulse
                  << " dL=" << current.angularImpulse - initial.angularImpulse
                  << " segmentDL=" << current.angularImpulse - segmentReference.angularImpulse;
    }
    std::cout << " removedPairs=" << removedPairs
              << " removedUpperPairs=" << removedUpperPairs
              << " reinjectedPairs=" << reinjectedPairs << '\n'
              << std::flush;
}
