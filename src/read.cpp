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
    const double parsedValue = std::stod(value, &parsed);
    if (parsed != value.size() || !std::isfinite(parsedValue))
        throw std::invalid_argument("expected a finite floating-point value");
    return parsedValue;
}
unsigned long long parseUnsigned(const std::string &value) {
    if (!value.empty() && value.front() == '-')
        throw std::invalid_argument("expected a non-negative integer");
    std::size_t parsed = 0;
    const auto parsedValue = std::stoull(value, &parsed);
    if (parsed != value.size())
        throw std::invalid_argument("expected an integer");
    return parsedValue;
}
int parseInt(const std::string &value) {
    std::size_t parsed = 0;
    const int parsedValue = std::stoi(value, &parsed);
    if (parsed != value.size())
        throw std::invalid_argument("expected an integer");
    return parsedValue;
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

struct ParameterParseState {
    std::optional<std::size_t> legacyNumSteps;
    bool hasExplicitEndTime = false;
};

bool assignCommonParameter(const std::string &key, const std::string &value, SimParams &params,
                           ParameterParseState &state) {
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
        {"dipoleRemovalUpperDistance", &SimParams::dipoleRemovalUpperDistance},
        {"dipoleRemovalInterval", &SimParams::dipoleRemovalInterval},
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
        {"initialConditionFile", &SimParams::initialConditionFile},
        {"restartFile", &SimParams::restartFile},
        {"runDirectory", &SimParams::runDirectory},
    };

    if (assignParameter(key, params, doubles, value, parseDouble)) {
        if (key == "endTime")
            state.hasExplicitEndTime = true;
        return true;
    }
    return assignParameter(key, params, optionalDoubles, value, parseDouble) ||
           assignParameter(key, params, integers, value, parseInt) ||
           assignParameter(key, params, strings, value,
                           [](const std::string &text) { return text; });
}

bool assignBooleanParameter(const std::string &key, const std::string &value,
                            SimParams &params) {
    if (key != "dipoleRemoval" && key != "dipoleRemovalUpper" && key != "overwriteRun")
        return false;

    const bool enabled = value == "true" || value == "1";
    if (!enabled && value != "false" && value != "0")
        throw std::invalid_argument(key + " must be true or false");

    if (key == "dipoleRemoval")
        params.dipoleRemoval = enabled;
    else if (key == "dipoleRemovalUpper")
        params.dipoleRemovalUpper = enabled;
    else
        params.overwriteRun = enabled;
    return true;
}

bool assignEnumParameter(const std::string &key, const std::string &value, SimParams &params) {
    if (key == "boundaryCondition") {
        const auto boundary = boundaryFromString(value);
        if (!boundary) {
            throw std::invalid_argument("boundaryCondition must be " +
                                        enumChoices<BoundaryKind>());
        }
        params.boundary = *boundary;
        return true;
    }
    if (key == "dipoleReinjection") {
        const auto mode = reinjectionFromString(value);
        if (!mode)
            throw std::invalid_argument("dipoleReinjection must be " +
                                        enumChoices<ReinjectionMode>());
        params.dipoleReinjection = *mode;
        return true;
    }
    if (key == "initialCondition") {
        const auto condition = initialConditionFromString(value);
        if (!condition) {
            throw std::invalid_argument("initialCondition must be " +
                                        enumChoices<InitialConditionKind>());
        }
        params.initialCondition = *condition;
        return true;
    }
    if (key == "integrator") {
        const auto integrator = integratorFromString(value);
        if (!integrator)
            throw std::invalid_argument("integrator must be " + enumChoices<IntegratorKind>());
        params.integrator = *integrator;
        return true;
    }
    return false;
}

bool assignLegacyParameter(const std::string &key, const std::string &value, SimParams &params,
                           ParameterParseState &state) {
    if (key == "N") {
        const auto count = parseUnsigned(value);
        if (count > std::numeric_limits<std::size_t>::max())
            throw std::out_of_range("N is too large");
        params.vortexCount = static_cast<std::size_t>(count);
        return true;
    }
    if (key == "numSteps") {
        state.legacyNumSteps = parseUnsigned(value);
        return true;
    }
    if (key == "coreSize") {
        params.coreRadius = std::sqrt(parseDouble(value));
        return true;
    }
    if (key == "randomSeed") {
        params.randomSeed = parseUnsigned(value);
        return true;
    }
    return false;
}

void assignParameterValue(const std::string &key, const std::string &value, SimParams &params,
                          ParameterParseState &state) {
    if (assignCommonParameter(key, value, params, state) ||
        assignBooleanParameter(key, value, params) || assignEnumParameter(key, value, params) ||
        assignLegacyParameter(key, value, params, state)) {
        return;
    }
    throw std::invalid_argument("unknown parameter: " + key);
}

void parseParameterLine(std::string line, std::size_t lineNumber, SimParams &params,
                        ParameterParseState &state) {
    const auto comment = line.find('#');
    if (comment != std::string::npos)
        line.erase(comment);

    std::istringstream fields(line);
    std::string key;
    if (!(fields >> key))
        return;

    std::string value;
    if (!(fields >> value))
        throw std::runtime_error("missing value on parameter line " + std::to_string(lineNumber));

    try {
        assignParameterValue(key, value, params, state);
        std::string trailing;
        if (fields >> trailing)
            throw std::invalid_argument("unexpected extra value: " + trailing);
    } catch (const std::exception &error) {
        throw std::runtime_error("parameter line " + std::to_string(lineNumber) + ": " +
                                 error.what());
    }
}

void validateNumericRanges(const SimParams &params) {
    for (double value : {params.timeStep, params.endTime, params.outputTime, params.coreRadius,
                         params.absoluteTolerance, params.relativeTolerance,
                         params.minimumTimeStep, params.maximumTimeStep, params.boxLengthX,
                         params.boxLengthY, params.diskRadius, params.dipoleRemovalDistance,
                         params.dipoleRemovalUpperDistance, params.dipoleRemovalInterval}) {
        if (!std::isfinite(value))
            throw std::invalid_argument("simulation parameters must be finite");
    }

    for (double radius : {params.coreRadius, params.diskRadius, params.dipoleRemovalDistance,
                          params.dipoleRemovalUpperDistance}) {
        if (!std::isfinite(radius * radius) || (radius > 0.0 && radius * radius == 0.0)) {
            throw std::invalid_argument(
                "radius or distance is outside the supported numeric range");
        }
    }
}

void validateTimeIntegration(const SimParams &params) {
    if (params.vortexCount == 0 && params.initialCondition != InitialConditionKind::file &&
        params.restartFile.empty())
        throw std::invalid_argument("N must be positive");
    if (!(params.timeStep > 0.0))
        throw std::invalid_argument("timeStep must be positive");
    if (!(params.endTime >= 0.0))
        throw std::invalid_argument("endTime must be non-negative");
    if (!(params.outputTime > 0.0))
        throw std::invalid_argument("outputTime must be positive");
    if (!std::isfinite(params.outputTime) ||
        !std::isfinite(params.diagnosticsInterval()) ||
        !(params.diagnosticsInterval() > 0.0) ||
        !std::isfinite(params.checkpointInterval()) ||
        !(params.checkpointInterval() > 0.0)) {
        throw std::invalid_argument("output intervals must be finite and positive");
    }
    if (!(params.coreRadius >= 0.0))
        throw std::invalid_argument("coreRadius must be non-negative");
    if (!(params.absoluteTolerance > 0.0) || !(params.relativeTolerance >= 0.0))
        throw std::invalid_argument("invalid integration tolerances");
    if (!(params.minimumTimeStep > 0.0) ||
        !(params.maximumTimeStep >= params.minimumTimeStep)) {
        throw std::invalid_argument("invalid timestep bounds");
    }
    if (params.numThreads < 0)
        throw std::invalid_argument("numThreads must be non-negative");
}

void validateGeometry(const SimParams &params) {
    switch (params.boundary) {
    case BoundaryKind::infinite:
    case BoundaryKind::periodic_x:
    case BoundaryKind::periodic:
    case BoundaryKind::disk:
        break;
    default:
        throw std::invalid_argument("invalid boundaryCondition");
    }
    if (!(params.boxLengthX > 0.0) || !(params.boxLengthY > 0.0) ||
        !(params.diskRadius > 0.0) || params.periodicImageLayers < 0 ||
        params.periodicImageLayers > 64) {
        throw std::invalid_argument("invalid geometry parameters or periodicImageLayers > 64");
    }
    if (params.boundary != BoundaryKind::infinite && params.coreRadius != 0.0) {
        throw std::invalid_argument("non-infinite geometries currently require coreRadius 0");
    }
}

void validateDipoleSettings(const SimParams &params) {
    if (!(params.dipoleRemovalDistance > 0.0))
        throw std::invalid_argument("dipoleRemovalDistance must be positive");
    if (!(params.dipoleRemovalUpperDistance > 0.0))
        throw std::invalid_argument("dipoleRemovalUpperDistance must be positive");
    if (params.dipoleRemovalUpper && !params.dipoleRemoval)
        throw std::invalid_argument("dipoleRemovalUpper requires dipoleRemoval true");
    if (params.dipoleRemovalUpper &&
        !(params.dipoleRemovalUpperDistance > params.dipoleRemovalDistance)) {
        throw std::invalid_argument(
            "dipoleRemovalUpperDistance must exceed dipoleRemovalDistance");
    }
    if (!(params.dipoleRemovalInterval >= 0.0))
        throw std::invalid_argument("dipoleRemovalInterval must be non-negative");
    if (!params.dipoleRemoval && params.dipoleReinjection != ReinjectionMode::none)
        throw std::invalid_argument("dipoleReinjection requires dipoleRemoval true");
    if (!isPeriodicY(params.boundary) && !isDisk(params.boundary) &&
        params.dipoleReinjection != ReinjectionMode::none) {
        throw std::invalid_argument(
            "dipole reinjection is available only for periodic and disk geometries");
    }
}

void validateInputAndOutput(const SimParams &params) {
    if (isPeriodicY(params.boundary) &&
        std::abs(params.boxLengthX - params.boxLengthY) >
            1e-13 * std::max(params.boxLengthX, params.boxLengthY)) {
        throw std::invalid_argument("Weiss-McWilliams periodic geometry requires a square box");
    }
    if (params.runDirectory.empty())
        throw std::invalid_argument("runDirectory must not be empty");
    if (params.initialCondition == InitialConditionKind::file &&
        params.initialConditionFile.empty()) {
        throw std::invalid_argument("initialCondition file requires initialConditionFile");
    }
    if (params.initialCondition != InitialConditionKind::file &&
        !params.initialConditionFile.empty()) {
        throw std::invalid_argument("initialConditionFile requires initialCondition file");
    }
}
} // namespace

void SimParams::validate() const {
    validateNumericRanges(*this);
    validateTimeIntegration(*this);
    validateGeometry(*this);
    validateDipoleSettings(*this);
    validateInputAndOutput(*this);
}
SimParams loadParams(const std::string &filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open parameter file: " + filename);
    SimParams params;
    ParameterParseState parseState;
    std::string line;
    std::size_t lineNumber = 0;
    // Strict parsing prevents a misspelled option from silently using a default.
    while (std::getline(input, line)) {
        ++lineNumber;
        parseParameterLine(line, lineNumber, params, parseState);
    }
    if (input.bad())
        throw std::runtime_error("failed while reading parameter file: " + filename);
    if (parseState.legacyNumSteps && !parseState.hasExplicitEndTime) {
        params.endTime = params.timeStep * static_cast<double>(*parseState.legacyNumSteps);
    }
    params.validate();
    return params;
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
