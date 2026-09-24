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
           "reinjected_pairs\n";
    output_.flush();
}
void DiagnosticsWriter::write(double time, std::size_t frame, const Invariants &value,
                              const Invariants &segmentReference, std::size_t removedPairs,
                              std::size_t reinjectedPairs) {
    output_ << time << ',' << frame << ',' << value.circulation << ',' << value.linearImpulseX
            << ',' << value.linearImpulseY << ',' << value.angularImpulse << ','
            << value.hamiltonian << ',' << value.circulation - initial_.circulation << ','
            << value.linearImpulseX - initial_.linearImpulseX << ','
            << value.linearImpulseY - initial_.linearImpulseY << ','
            << value.angularImpulse - initial_.angularImpulse << ','
            << value.hamiltonian - initial_.hamiltonian << ','
            << value.circulation - segmentReference.circulation << ','
            << value.linearImpulseX - segmentReference.linearImpulseX << ','
            << value.linearImpulseY - segmentReference.linearImpulseY << ','
            << value.angularImpulse - segmentReference.angularImpulse << ','
            << value.hamiltonian - segmentReference.hamiltonian << ',' << removedPairs << ','
            << reinjectedPairs << '\n';
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
           << "boundaryCondition " << params.boundaryCondition << '\n'
           << "boxLengthX " << params.boxLengthX << '\n'
           << "boxLengthY " << params.boxLengthY << '\n'
           << "periodicImageLayers " << params.periodicImageLayers << '\n'
           << "diskRadius " << params.diskRadius << '\n'
           << "randomSeed " << params.randomSeed << '\n'
           << "dipoleRemoval " << params.dipoleRemoval << '\n'
           << "dipoleRemovalDistance " << params.dipoleRemovalDistance << '\n'
           << "dipoleReinjection " << toString(params.dipoleReinjection) << '\n'
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
void printDiagnostics(double time, std::size_t steps, const Invariants &value,
                      const Invariants &initial, const std::string &boundaryCondition,
                      const Invariants &segmentReference, std::size_t removedPairs,
                      std::size_t reinjectedPairs) {
    std::cout << std::setprecision(10) << "time=" << time << " steps=" << steps
              << " circulation=" << value.circulation
              << " dCirculation=" << value.circulation - initial.circulation
              << " segmentDCirculation=" << value.circulation - segmentReference.circulation
              << " H=" << value.hamiltonian << " dH=" << value.hamiltonian - initial.hamiltonian
              << " segmentDH=" << value.hamiltonian - segmentReference.hamiltonian;

    if (boundaryCondition == "infinite" || boundaryCondition == "periodic_x" ||
        boundaryCondition == "periodic") {
        std::cout << " Ix=" << value.linearImpulseX
                  << " dIx=" << value.linearImpulseX - initial.linearImpulseX
                  << " segmentDIx=" << value.linearImpulseX - segmentReference.linearImpulseX
                  << " Iy=" << value.linearImpulseY
                  << " dIy=" << value.linearImpulseY - initial.linearImpulseY
                  << " segmentDIy=" << value.linearImpulseY - segmentReference.linearImpulseY;
    }
    if (boundaryCondition == "infinite" || boundaryCondition == "disk") {
        std::cout << " L=" << value.angularImpulse
                  << " dL=" << value.angularImpulse - initial.angularImpulse
                  << " segmentDL=" << value.angularImpulse - segmentReference.angularImpulse;
    }
    std::cout << " removedPairs=" << removedPairs << " reinjectedPairs=" << reinjectedPairs << '\n'
              << std::flush;
}
