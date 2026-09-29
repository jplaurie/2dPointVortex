#ifndef POINT_VORTEX_PARAMS_H
#define POINT_VORTEX_PARAMS_H
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
enum class IntegratorKind : std::uint8_t { rk4, dopri5 };
enum class ReinjectionMode : std::uint8_t { none, independent, paired };
enum class InitialConditionKind : std::uint8_t { random, ring, single, dipole, file };
enum class BoundaryKind : std::uint8_t { infinite, periodic_x, periodic, disk };
[[nodiscard]] constexpr const char *toString(IntegratorKind kind) noexcept {
    return kind == IntegratorKind::rk4 ? "rk4" : "dopri5";
}
[[nodiscard]] constexpr const char *toString(ReinjectionMode mode) noexcept {
    switch (mode) {
    case ReinjectionMode::independent:
        return "independent";
    case ReinjectionMode::paired:
        return "paired";
    case ReinjectionMode::none:
        return "none";
    }
    return "none";
}
[[nodiscard]] constexpr const char *toString(InitialConditionKind kind) noexcept {
    switch (kind) {
    case InitialConditionKind::random:
        return "random";
    case InitialConditionKind::ring:
        return "ring";
    case InitialConditionKind::single:
        return "single";
    case InitialConditionKind::dipole:
        return "dipole";
    case InitialConditionKind::file:
        return "file";
    }
    return "unknown";
}
[[nodiscard]] constexpr const char *toString(BoundaryKind kind) noexcept {
    switch (kind) {
    case BoundaryKind::infinite:
        return "infinite";
    case BoundaryKind::periodic_x:
        return "periodic_x";
    case BoundaryKind::periodic:
        return "periodic";
    case BoundaryKind::disk:
        return "disk";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::optional<IntegratorKind>
integratorFromString(std::string_view value) noexcept {
    if (value == "rk4")
        return IntegratorKind::rk4;
    if (value == "dopri5")
        return IntegratorKind::dopri5;
    return std::nullopt;
}
[[nodiscard]] constexpr std::optional<ReinjectionMode>
reinjectionFromString(std::string_view value) noexcept {
    if (value == "none")
        return ReinjectionMode::none;
    if (value == "independent")
        return ReinjectionMode::independent;
    if (value == "paired")
        return ReinjectionMode::paired;
    return std::nullopt;
}
[[nodiscard]] constexpr std::optional<InitialConditionKind>
initialConditionFromString(std::string_view value) noexcept {
    if (value == "random")
        return InitialConditionKind::random;
    if (value == "ring")
        return InitialConditionKind::ring;
    if (value == "single")
        return InitialConditionKind::single;
    if (value == "dipole")
        return InitialConditionKind::dipole;
    if (value == "file")
        return InitialConditionKind::file;
    return std::nullopt;
}
[[nodiscard]] constexpr std::optional<BoundaryKind>
boundaryFromString(std::string_view value) noexcept {
    if (value == "infinite")
        return BoundaryKind::infinite;
    if (value == "periodic_x")
        return BoundaryKind::periodic_x;
    if (value == "periodic")
        return BoundaryKind::periodic;
    if (value == "disk")
        return BoundaryKind::disk;
    return std::nullopt;
}
struct SimParams {
    // Simulation and integrator controls.
    std::size_t vortexCount = 100;
    double timeStep = 1.0e-3;
    double endTime = 1.0;
    double outputTime = 0.1;
    // Unset intervals follow outputTime, preserving existing parameter files.
    std::optional<double> diagnosticsTime;
    std::optional<double> checkpointTime;
    [[nodiscard]] double diagnosticsInterval() const {
        return diagnosticsTime.value_or(outputTime);
    }
    [[nodiscard]] double checkpointInterval() const { return checkpointTime.value_or(outputTime); }
    double coreRadius = 0.0;
    double absoluteTolerance = 1.0e-10;
    double relativeTolerance = 1.0e-8;
    double minimumTimeStep = 1.0e-12;
    double maximumTimeStep = 0.1;
    int numThreads = 0;
    IntegratorKind integrator = IntegratorKind::dopri5;

    // Geometry controls. Only the fields selected by boundary are active.
    BoundaryKind boundary = BoundaryKind::infinite;
    // periodic_x uses boxLengthX; periodic uses both box lengths and image layers.
    double boxLengthX = 2.0;
    double boxLengthY = 2.0;
    double diskRadius = 1.0;
    int periodicImageLayers = 8;
    std::uint64_t randomSeed = 1234567;

    // Optional lower/upper dipole removal and geometry-aware reinjection.
    bool dipoleRemoval = false;
    double dipoleRemovalDistance = 0.01;
    bool dipoleRemovalUpper = false;
    double dipoleRemovalUpperDistance = 1.0;
    // Zero processes after every accepted step; positive values use a time cadence.
    double dipoleRemovalInterval = 0.0;
    ReinjectionMode dipoleReinjection = ReinjectionMode::none;

    // Input, restart, and managed-run paths.
    InitialConditionKind initialCondition = InitialConditionKind::ring;
    std::string initialConditionFile;
    std::string restartFile;
    std::string runDirectory = "runs/default";
    // Replaces the managed solver artefacts in runDirectory when true.
    bool overwriteRun = false;
    void validate() const;
};

struct RunPaths {
    std::filesystem::path directory;
    std::filesystem::path trajectory;
    std::filesystem::path diagnostics;
    std::filesystem::path checkpoints;
};

[[nodiscard]] inline RunPaths runPaths(const SimParams &params) {
    const std::filesystem::path directory(params.runDirectory);
    return {directory, directory / "trajectory.csv", directory / "diagnostics.csv",
            directory / "checkpoints"};
}
#endif
