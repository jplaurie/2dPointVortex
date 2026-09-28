#ifndef POINT_VORTEX_PRINT_H
#define POINT_VORTEX_PRINT_H
#include "compute.h"
#include "params.h"
#include "vortex.h"
#include <fstream>
#include <string>
class TrajectoryWriter {
  public:
    TrajectoryWriter(const std::string &filename, bool overwrite);
    void write(double time, std::size_t frame, const VortexSystem &vortices,
               const VelocityField &velocity);

  private:
    std::ofstream output_;
};
class DiagnosticsWriter {
  public:
    DiagnosticsWriter(const std::string &filename, const Invariants &initial, bool overwrite);
    void write(double time, std::size_t frame, const Invariants &values,
               const Invariants &segmentReference, std::size_t removedPairs,
               std::size_t removedUpperPairs, std::size_t reinjectedPairs);

  private:
    std::ofstream output_;
    Invariants initial_;
};
void writeRunProvenance(const SimParams &params, const std::string &parameterFile,
                        const std::string &backend, const std::string &runtimeDetails,
                        double startTime, std::size_t startFrame, bool restarting);
void printDiagnostics(double time, std::size_t steps, const Invariants &current,
                      const Invariants &initial, const std::string &boundaryCondition,
                      const Invariants &segmentReference, std::size_t removedPairs,
                      std::size_t removedUpperPairs, std::size_t reinjectedPairs);
#endif
