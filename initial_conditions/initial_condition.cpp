#include "initial_condition.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {

void validateOptions(const InitialConditionOptions &options) {
    if (options.count == 0)
        throw std::invalid_argument("vortex count must be positive");
    for (double value :
         {options.circulationMagnitude, options.minimumSeparation, options.infiniteHalfWidth,
          options.boxLength, options.diskRadius, options.ringRadius})
        if (!std::isfinite(value))
            throw std::invalid_argument("initial-condition options must be finite");
    if (!(options.circulationMagnitude > 0.0) || !(options.minimumSeparation >= 0.0) ||
        !(options.infiniteHalfWidth > 0.0) || !(options.boxLength > 0.0) ||
        !(options.diskRadius > 0.0) || !(options.ringRadius >= 0.0))
        throw std::invalid_argument("invalid initial-condition option");
    if (options.geometry == InitialGeometry::periodic &&
        ((options.pattern == InitialPattern::random || options.pattern == InitialPattern::ring)
             ? options.count % 2 != 0
             : options.pattern != InitialPattern::dipole))
        throw std::invalid_argument("periodic initial condition must have zero circulation");
}

double distance(const VortexSystem &vortices, std::size_t first, std::size_t second,
                const InitialConditionOptions &options) {
    double dx = vortices.x[first] - vortices.x[second];
    double dy = vortices.y[first] - vortices.y[second];
    if (options.geometry == InitialGeometry::periodic ||
        options.geometry == InitialGeometry::periodic_x) {
        dx = std::remainder(dx, options.boxLength);
    }
    if (options.geometry == InitialGeometry::periodic) {
        dy = std::remainder(dy, options.boxLength);
    }
    return std::hypot(dx, dy);
}

double defaultPatternRadius(const InitialConditionOptions &options) {
    if (options.ringRadius > 0.0)
        return options.ringRadius;
    if (options.geometry == InitialGeometry::disk)
        return 0.5 * options.diskRadius;
    if (options.geometry == InitialGeometry::periodic ||
        options.geometry == InitialGeometry::periodic_x)
        return 0.25 * options.boxLength;
    return 0.5 * options.infiniteHalfWidth;
}

void assignAlternatingCirculation(VortexSystem &vortices, double magnitude) {
    for (std::size_t i = 0; i < vortices.size(); ++i)
        vortices.circulation[i] = i % 2 == 0 ? magnitude : -magnitude;
}

VortexSystem makeRandom(const InitialConditionOptions &options) {
    VortexSystem vortices(options.count);
    assignAlternatingCirculation(vortices, options.circulationMagnitude);
    std::mt19937_64 generator(options.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);
    constexpr std::size_t maximumAttemptsPerVortex = 1'000'000;

    for (std::size_t i = 0; i < vortices.size(); ++i) {
        bool accepted = false;
        for (std::size_t attempt = 0; attempt < maximumAttemptsPerVortex; ++attempt) {
            if (options.geometry == InitialGeometry::disk) {
                const double radius = options.diskRadius * std::sqrt(unit(generator));
                const double theta = angle(generator);
                vortices.x[i] = radius * std::cos(theta);
                vortices.y[i] = radius * std::sin(theta);
            } else {
                const bool periodicX = options.geometry == InitialGeometry::periodic ||
                                       options.geometry == InitialGeometry::periodic_x;
                const double halfWidthX =
                    periodicX ? 0.5 * options.boxLength : options.infiniteHalfWidth;
                const double halfWidthY = options.geometry == InitialGeometry::periodic
                                              ? 0.5 * options.boxLength
                                              : options.infiniteHalfWidth;
                vortices.x[i] = (2.0 * unit(generator) - 1.0) * halfWidthX;
                vortices.y[i] = (2.0 * unit(generator) - 1.0) * halfWidthY;
            }
            accepted = true;
            for (std::size_t j = 0; j < i; ++j)
                if (distance(vortices, i, j, options) < options.minimumSeparation) {
                    accepted = false;
                    break;
                }
            if (accepted)
                break;
        }
        if (!accepted)
            throw std::runtime_error(
                "could not place all vortices; reduce count or minimum separation");
    }
    return vortices;
}

VortexSystem makeRing(const InitialConditionOptions &options) {
    VortexSystem vortices(options.count);
    const double radius = defaultPatternRadius(options);
    for (std::size_t i = 0; i < vortices.size(); ++i) {
        const double angle =
            2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(vortices.size());
        vortices.x[i] = radius * std::cos(angle);
        vortices.y[i] = radius * std::sin(angle);
    }
    assignAlternatingCirculation(vortices, options.circulationMagnitude);
    return vortices;
}

VortexSystem makeFixedPattern(const InitialConditionOptions &options) {
    if (options.pattern == InitialPattern::single) {
        VortexSystem vortices(1);
        vortices.circulation[0] = options.circulationMagnitude;
        return vortices;
    }
    VortexSystem vortices(2);
    const double offset = 0.5 * defaultPatternRadius(options);
    vortices.x = {-offset, offset};
    vortices.y = {0.0, 0.0};
    vortices.circulation[0] = options.circulationMagnitude;
    vortices.circulation[1] = options.pattern == InitialPattern::dipole
                                  ? -options.circulationMagnitude
                                  : options.circulationMagnitude;
    return vortices;
}

} // namespace

const char *toString(InitialGeometry geometry) {
    switch (geometry) {
    case InitialGeometry::infinite:
        return "infinite";
    case InitialGeometry::periodic_x:
        return "periodic_x";
    case InitialGeometry::periodic:
        return "periodic";
    case InitialGeometry::disk:
        return "disk";
    }
    return "unknown";
}

const char *toString(InitialPattern pattern) {
    switch (pattern) {
    case InitialPattern::random:
        return "random";
    case InitialPattern::single:
        return "single";
    case InitialPattern::corotatingPair:
        return "pair";
    case InitialPattern::dipole:
        return "dipole";
    case InitialPattern::ring:
        return "ring";
    }
    return "unknown";
}

double minimumVortexSeparation(const VortexSystem &vortices,
                               const InitialConditionOptions &options) {
    if (vortices.size() < 2)
        return std::numeric_limits<double>::infinity();
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < vortices.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            minimum = std::min(minimum, distance(vortices, i, j, options));
    return minimum;
}

void validateInitialCondition(const VortexSystem &vortices,
                              const InitialConditionOptions &options) {
    vortices.validate();
    if (vortices.size() == 0)
        throw std::invalid_argument("vortex count must be positive");
    validateOptions(options);

    double totalCirculation = 0.0;
    for (std::size_t i = 0; i < vortices.size(); ++i) {
        if (vortices.circulation[i] == 0.0)
            throw std::invalid_argument("initial condition contains an invalid vortex");
        if (options.geometry == InitialGeometry::periodic &&
            (vortices.x[i] < -0.5 * options.boxLength || vortices.x[i] >= 0.5 * options.boxLength ||
             vortices.y[i] < -0.5 * options.boxLength || vortices.y[i] >= 0.5 * options.boxLength))
            throw std::invalid_argument("periodic vortex lies outside the fundamental box");
        if (options.geometry == InitialGeometry::periodic_x &&
            (vortices.x[i] < -0.5 * options.boxLength || vortices.x[i] >= 0.5 * options.boxLength))
            throw std::invalid_argument(
                "singly periodic vortex lies outside the fundamental x interval");
        if (options.geometry == InitialGeometry::disk &&
            vortices.x[i] * vortices.x[i] + vortices.y[i] * vortices.y[i] >=
                options.diskRadius * options.diskRadius)
            throw std::invalid_argument("disk vortex lies on or outside the boundary");
        totalCirculation += vortices.circulation[i];
    }
    if (options.geometry == InitialGeometry::periodic &&
        std::abs(totalCirculation) >
            1e-12 * static_cast<double>(vortices.size()) * options.circulationMagnitude)
        throw std::invalid_argument("periodic initial condition must have zero circulation");
    if (minimumVortexSeparation(vortices, options) < options.minimumSeparation)
        throw std::invalid_argument("initial condition violates minimum separation");
}

VortexSystem generateInitialCondition(const InitialConditionOptions &options) {
    validateOptions(options);
    VortexSystem vortices;
    if (options.pattern == InitialPattern::random)
        vortices = makeRandom(options);
    else if (options.pattern == InitialPattern::ring)
        vortices = makeRing(options);
    else
        vortices = makeFixedPattern(options);
    validateInitialCondition(vortices, options);
    return vortices;
}

void writeInitialCondition(const std::string &filename, const VortexSystem &vortices,
                           const InitialConditionOptions &options, bool overwrite) {
    validateInitialCondition(vortices, options);
    if (!overwrite && std::filesystem::exists(filename))
        throw std::runtime_error("refusing to overwrite initial-condition file: " + filename);
    const auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream output(filename, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write initial-condition file: " + filename);
    output << "# PointVortex initial condition\n"
           << "# geometry=" << toString(options.geometry)
           << " pattern=" << toString(options.pattern) << " seed=" << options.seed << '\n'
           << "# box_length=" << std::setprecision(17) << options.boxLength
           << " disk_radius=" << options.diskRadius
           << " infinite_half_width=" << options.infiniteHalfWidth << '\n'
           << "# minimum_separation=" << std::setprecision(17)
           << minimumVortexSeparation(vortices, options) << '\n'
           << "# x y circulation\n";
    for (std::size_t i = 0; i < vortices.size(); ++i)
        output << std::setprecision(17) << vortices.x[i] << ' ' << vortices.y[i] << ' '
               << vortices.circulation[i] << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("failed while writing initial-condition file: " + filename);
}
