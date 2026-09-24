#include "read.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
double parseDouble(const std::string &value) {
    std::size_t parsed = 0;
    const double result = std::stod(value, &parsed);
    if (parsed != value.size() || !std::isfinite(result))
        throw std::invalid_argument("expected a finite floating-point value");
    return result;
}
unsigned long long parseUnsigned(const std::string &value) {
    if (!value.empty() && value.front() == '-')
        throw std::invalid_argument("expected a non-negative integer");
    std::size_t parsed = 0;
    const auto result = std::stoull(value, &parsed);
    if (parsed != value.size())
        throw std::invalid_argument("expected an integer");
    return result;
}
int parseInt(const std::string &value) {
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size())
        throw std::invalid_argument("expected an integer");
    return result;
}

template <typename T> struct Parameter {
    std::string_view name;
    T SimParams::*member;
};

template <typename T, std::size_t N, typename Parser>
bool assignParameter(const std::string &key, SimParams &params,
                     const Parameter<T> (&parameters)[N], const std::string &value, Parser parser) {
    for (const auto &[name, member] : parameters)
        if (key == name) {
            params.*member = parser(value);
            return true;
        }
    return false;
}
} // namespace

void SimParams::validate() const {
    for (double value : {timeStep, endTime, outputTime, coreRadius, absoluteTolerance,
                         relativeTolerance, minimumTimeStep, maximumTimeStep, boxLengthX,
                         boxLengthY, diskRadius, dipoleRemovalDistance})
        if (!std::isfinite(value))
            throw std::invalid_argument("simulation parameters must be finite");
    for (double radius : {coreRadius, diskRadius, dipoleRemovalDistance})
        if (!std::isfinite(radius * radius) || (radius > 0.0 && radius * radius == 0.0))
            throw std::invalid_argument(
                "radius or distance is outside the supported numeric range");
    if (vortexCount == 0 && initialConditionFile.empty() && restartFile.empty())
        throw std::invalid_argument("N must be positive");
    if (!(timeStep > 0.0))
        throw std::invalid_argument("timeStep must be positive");
    if (!(endTime >= 0.0))
        throw std::invalid_argument("endTime must be non-negative");
    if (!(outputTime > 0.0))
        throw std::invalid_argument("outputTime must be positive");
    if (!std::isfinite(outputTime) || !std::isfinite(diagnosticsInterval()) ||
        !(diagnosticsInterval() > 0.0) || !std::isfinite(checkpointInterval()) ||
        !(checkpointInterval() > 0.0))
        throw std::invalid_argument("output intervals must be finite and positive");
    if (!(coreRadius >= 0.0))
        throw std::invalid_argument("coreRadius must be non-negative");
    if (!(absoluteTolerance > 0.0) || !(relativeTolerance >= 0.0))
        throw std::invalid_argument("invalid integration tolerances");
    if (!(minimumTimeStep > 0.0) || !(maximumTimeStep >= minimumTimeStep))
        throw std::invalid_argument("invalid timestep bounds");
    if (numThreads < 0)
        throw std::invalid_argument("numThreads must be non-negative");
    if (boundaryCondition != "infinite" && boundaryCondition != "periodic_x" &&
        boundaryCondition != "periodic" && boundaryCondition != "disk")
        throw std::invalid_argument("invalid boundaryCondition");
    if (!(boxLengthX > 0.0) || !(boxLengthY > 0.0) || !(diskRadius > 0.0) ||
        periodicImageLayers < 0 || periodicImageLayers > 64)
        throw std::invalid_argument("invalid geometry parameters or periodicImageLayers > 64");
    if (boundaryCondition != "infinite" && coreRadius != 0.0)
        throw std::invalid_argument("non-infinite geometries currently require coreRadius 0");
    if (!(dipoleRemovalDistance > 0.0))
        throw std::invalid_argument("dipoleRemovalDistance must be positive");
    if (!dipoleRemoval && dipoleReinjection != ReinjectionMode::none)
        throw std::invalid_argument("dipoleReinjection requires dipoleRemoval true");
    if (boundaryCondition != "periodic" && boundaryCondition != "disk" &&
        dipoleReinjection != ReinjectionMode::none)
        throw std::invalid_argument(
            "dipole reinjection is available only for periodic and disk geometries");
    if (boundaryCondition == "periodic" &&
        std::abs(boxLengthX - boxLengthY) > 1e-13 * std::max(boxLengthX, boxLengthY))
        throw std::invalid_argument("Weiss-McWilliams periodic geometry requires a square box");
    if (runDirectory.empty())
        throw std::invalid_argument("runDirectory must not be empty");
}
SimParams loadParams(const std::string &filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open parameter file: " + filename);
    SimParams p;
    std::optional<std::size_t> legacyNumSteps;
    bool explicitEndTime = false;
    static constexpr Parameter<double> doubles[] = {
        {"timeStep", &SimParams::timeStep},
        {"endTime", &SimParams::endTime},
        {"outputTime", &SimParams::outputTime},
        {"OutputTime", &SimParams::outputTime},
        {"coreRadius", &SimParams::coreRadius},
        {"absoluteTolerance", &SimParams::absoluteTolerance},
        {"relativeTolerance", &SimParams::relativeTolerance},
        {"minimumTimeStep", &SimParams::minimumTimeStep},
        {"maximumTimeStep", &SimParams::maximumTimeStep},
        {"boxLengthX", &SimParams::boxLengthX},
        {"boxLengthY", &SimParams::boxLengthY},
        {"diskRadius", &SimParams::diskRadius},
        {"dipoleRemovalDistance", &SimParams::dipoleRemovalDistance},
    };
    static constexpr Parameter<std::optional<double>> optionalDoubles[] = {
        {"diagnosticsTime", &SimParams::diagnosticsTime},
        {"checkpointTime", &SimParams::checkpointTime},
    };
    static constexpr Parameter<int> integers[] = {
        {"numThreads", &SimParams::numThreads},
        {"periodicImageLayers", &SimParams::periodicImageLayers},
    };
    static constexpr Parameter<std::string> strings[] = {
        {"boundaryCondition", &SimParams::boundaryCondition},
        {"initialConditionFile", &SimParams::initialConditionFile},
        {"restartFile", &SimParams::restartFile},
        {"runDirectory", &SimParams::runDirectory},
    };
    std::string line;
    std::size_t lineNumber = 0;
    // Strict parsing prevents a misspelled option from silently using a default.
    while (std::getline(input, line)) {
        ++lineNumber;
        const auto comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);
        std::istringstream fields(line);
        std::string key, value;
        if (!(fields >> key))
            continue;
        if (!(fields >> value))
            throw std::runtime_error("missing value on parameter line " +
                                     std::to_string(lineNumber));
        try {
            if (assignParameter(key, p, doubles, value, parseDouble)) {
                if (key == "endTime")
                    explicitEndTime = true;
            } else if (assignParameter(key, p, optionalDoubles, value, parseDouble) ||
                       assignParameter(key, p, integers, value, parseInt) ||
                       assignParameter(key, p, strings, value,
                                       [](const std::string &text) { return text; })) {
            } else if (key == "N") {
                const auto count = parseUnsigned(value);
                if (count > std::numeric_limits<std::size_t>::max())
                    throw std::out_of_range("N is too large");
                p.vortexCount = static_cast<std::size_t>(count);
            } else if (key == "numSteps")
                legacyNumSteps = parseUnsigned(value);
            else if (key == "coreSize")
                p.coreRadius = std::sqrt(parseDouble(value));
            else if (key == "randomSeed")
                p.randomSeed = parseUnsigned(value);
            else if (key == "dipoleRemoval" || key == "overwriteRun") {
                const bool enabled = value == "true" || value == "1";
                if (!enabled && value != "false" && value != "0")
                    throw std::invalid_argument(key + " must be true or false");
                if (key == "dipoleRemoval")
                    p.dipoleRemoval = enabled;
                else
                    p.overwriteRun = enabled;
            } else if (key == "dipoleReinjection") {
                const auto mode = reinjectionFromString(value);
                if (!mode)
                    throw std::invalid_argument(
                        "dipoleReinjection must be none, independent, or paired");
                p.dipoleReinjection = *mode;
            } else if (key == "integrator") {
                const auto integrator = integratorFromString(value);
                if (!integrator)
                    throw std::invalid_argument("integrator must be rk4 or dopri5");
                p.integrator = *integrator;
            } else
                throw std::invalid_argument("unknown parameter: " + key);
            std::string trailing;
            if (fields >> trailing)
                throw std::invalid_argument("unexpected extra value: " + trailing);
        } catch (const std::exception &e) {
            throw std::runtime_error("parameter line " + std::to_string(lineNumber) + ": " +
                                     e.what());
        }
    }
    if (input.bad())
        throw std::runtime_error("failed while reading parameter file: " + filename);
    if (legacyNumSteps && !explicitEndTime)
        p.endTime = p.timeStep * static_cast<double>(*legacyNumSteps);
    p.validate();
    return p;
}
VortexSystem loadVortices(const std::string &filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open initial-condition file: " + filename);
    VortexSystem vortices;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        const auto comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);
        for (char &character : line)
            if (character == ',')
                character = ' ';
        std::istringstream fields(line);
        double x, y, circulation;
        fields >> std::ws;
        if (fields.eof())
            continue;
        if (!(fields >> x >> y >> circulation))
            throw std::runtime_error("invalid initial condition on line " +
                                     std::to_string(lineNumber));
        std::string trailing;
        if (fields >> trailing || !std::isfinite(x) || !std::isfinite(y) ||
            !std::isfinite(circulation))
            throw std::runtime_error("invalid initial condition on line " +
                                     std::to_string(lineNumber));
        vortices.pushBack(x, y, circulation);
    }
    if (input.bad())
        throw std::runtime_error("failed while reading initial-condition file: " + filename);
    if (vortices.size() == 0)
        throw std::runtime_error("initial-condition file is empty");
    return vortices;
}
InitialConditionMetadata readInitialConditionMetadata(const std::string &filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open initial-condition file: " + filename);
    InitialConditionMetadata metadata;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment == std::string::npos)
            continue;
        std::istringstream fields(line.substr(comment + 1));
        std::string field;
        while (fields >> field) {
            const auto equals = field.find('=');
            if (equals == std::string::npos)
                continue;
            const std::string key = field.substr(0, equals);
            const std::string value = field.substr(equals + 1);
            try {
                if (key == "geometry")
                    metadata.geometry = value;
                else if (key == "box_length")
                    metadata.boxLength = parseDouble(value);
                else if (key == "disk_radius")
                    metadata.diskRadius = parseDouble(value);
            } catch (const std::exception &error) {
                throw std::runtime_error("invalid initial-condition metadata: " +
                                         std::string(error.what()));
            }
        }
    }
    return metadata;
}
void initializeVortices(VortexSystem &vortices, double radius) {
    for (std::size_t i = 0; i < vortices.size(); ++i) {
        const double angle =
            2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(vortices.size());
        vortices.x[i] = radius * std::cos(angle);
        vortices.y[i] = radius * std::sin(angle);
        vortices.circulation[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
}

void initializePeriodicVortices(VortexSystem &vortices, double lengthX, double lengthY,
                                std::uint64_t seed) {
    if (vortices.size() % 2 != 0)
        throw std::invalid_argument("the built-in periodic initial condition requires an even N");

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> xPosition(-0.5 * lengthX, 0.5 * lengthX);
    std::uniform_real_distribution<double> yPosition(-0.5 * lengthY, 0.5 * lengthY);
    for (std::size_t i = 0; i < vortices.size(); ++i) {
        vortices.x[i] = xPosition(generator);
        vortices.y[i] = yPosition(generator);
        // Equal numbers of positive and negative vortices satisfy periodic neutrality.
        vortices.circulation[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
}
