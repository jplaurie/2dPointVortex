#ifndef POINT_VORTEX_PARAMS_H
#define POINT_VORTEX_PARAMS_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
enum class IntegratorKind : std::uint8_t { rk4, dopri5 };
enum class ReinjectionMode : std::uint8_t { none, independent, paired };
enum class InitialConditionKind : std::uint8_t { random, ring, single, dipole, file };
enum class BoundaryKind : std::uint8_t { infinite, periodic_x, periodic, disk };

namespace parameter_detail {
template <typename Enum> struct EnumTraits;

template <> struct EnumTraits<IntegratorKind> {
    static constexpr std::array names{
        std::pair{IntegratorKind::rk4, "rk4"},
        std::pair{IntegratorKind::dopri5, "dopri5"},
    };
};
template <> struct EnumTraits<ReinjectionMode> {
    static constexpr std::array names{
        std::pair{ReinjectionMode::none, "none"},
        std::pair{ReinjectionMode::independent, "independent"},
        std::pair{ReinjectionMode::paired, "paired"},
    };
};
template <> struct EnumTraits<InitialConditionKind> {
    static constexpr std::array names{
        std::pair{InitialConditionKind::random, "random"},
        std::pair{InitialConditionKind::ring, "ring"},
        std::pair{InitialConditionKind::single, "single"},
        std::pair{InitialConditionKind::dipole, "dipole"},
        std::pair{InitialConditionKind::file, "file"},
    };
};
template <> struct EnumTraits<BoundaryKind> {
    static constexpr std::array names{
        std::pair{BoundaryKind::infinite, "infinite"},
        std::pair{BoundaryKind::periodic_x, "periodic_x"},
        std::pair{BoundaryKind::periodic, "periodic"},
        std::pair{BoundaryKind::disk, "disk"},
    };
};

template <typename Enum> [[nodiscard]] constexpr const char *enumName(Enum value) noexcept {
    for (const auto &[candidate, name] : EnumTraits<Enum>::names)
        if (candidate == value)
            return name;
    return "unknown";
}

template <typename Enum>
[[nodiscard]] constexpr std::optional<Enum> enumFromString(std::string_view value) noexcept {
    for (const auto &[candidate, name] : EnumTraits<Enum>::names)
        if (value == name)
            return candidate;
    return std::nullopt;
}
} // namespace parameter_detail

template <typename Enum> [[nodiscard]] std::string enumChoices() {
    const auto &names = parameter_detail::EnumTraits<Enum>::names;
    std::string choices;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index > 0)
            choices += index + 1 == names.size() ? (names.size() == 2 ? " or " : ", or ") : ", ";
        choices += names[index].second;
    }
    return choices;
}

[[nodiscard]] constexpr const char *toString(IntegratorKind kind) noexcept {
    return parameter_detail::enumName(kind);
}
[[nodiscard]] constexpr const char *toString(ReinjectionMode mode) noexcept {
    return parameter_detail::enumName(mode);
}
[[nodiscard]] constexpr const char *toString(InitialConditionKind kind) noexcept {
    return parameter_detail::enumName(kind);
}
[[nodiscard]] constexpr const char *toString(BoundaryKind kind) noexcept {
    return parameter_detail::enumName(kind);
}
[[nodiscard]] constexpr std::optional<IntegratorKind>
integratorFromString(std::string_view value) noexcept {
    return parameter_detail::enumFromString<IntegratorKind>(value);
}
[[nodiscard]] constexpr std::optional<ReinjectionMode>
reinjectionFromString(std::string_view value) noexcept {
    return parameter_detail::enumFromString<ReinjectionMode>(value);
}
[[nodiscard]] constexpr std::optional<InitialConditionKind>
initialConditionFromString(std::string_view value) noexcept {
    return parameter_detail::enumFromString<InitialConditionKind>(value);
}
[[nodiscard]] constexpr std::optional<BoundaryKind>
boundaryFromString(std::string_view value) noexcept {
    return parameter_detail::enumFromString<BoundaryKind>(value);
}

[[nodiscard]] constexpr bool isPeriodicX(BoundaryKind boundary) noexcept {
    return boundary == BoundaryKind::periodic_x || boundary == BoundaryKind::periodic;
}
[[nodiscard]] constexpr bool isPeriodicY(BoundaryKind boundary) noexcept {
    return boundary == BoundaryKind::periodic;
}
[[nodiscard]] constexpr bool isDisk(BoundaryKind boundary) noexcept {
    return boundary == BoundaryKind::disk;
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

struct GeometrySignature {
    double lengthX;
    double lengthY;
    int imageLayers;
};

[[nodiscard]] inline GeometrySignature geometrySignature(const SimParams &params) {
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
