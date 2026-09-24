#include "dipole.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
struct Candidate {
    std::size_t first;
    std::size_t second;
    double distanceSquared;
};

struct RemovedEvent {
    double firstCirculation;
    double secondCirculation = 0.0;
    bool wallImage = false;
};

double displacement(double difference, double length, bool periodic) {
    return periodic ? std::remainder(difference, length) : difference;
}
} // namespace

DipoleManager::DipoleManager(const SimParams &params)
    : params_(params), random_(params.randomSeed ^ 0xd1b54a32d192ed03ULL) {}

DipoleManager::DipoleManager(const SimParams &params, const DipoleEventState &state)
    : params_(params), random_(params.randomSeed ^ 0xd1b54a32d192ed03ULL),
      removedPairs_(state.removedPairs), reinjectedPairs_(state.reinjectedPairs) {
    if (!state.randomEngineState.empty()) {
        std::istringstream input(state.randomEngineState);
        if (!(input >> random_))
            throw std::runtime_error("invalid random-generator state in checkpoint");
    }
}

std::size_t DipoleManager::process(VortexSystem &vortices) {
    if (!params_.dipoleRemoval || vortices.size() == 0)
        return 0;

    const bool periodicX =
        params_.boundaryCondition == "periodic" || params_.boundaryCondition == "periodic_x";
    const bool periodicY = params_.boundaryCondition == "periodic";
    const double thresholdSquared = params_.dipoleRemovalDistance * params_.dipoleRemovalDistance;
    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < vortices.size(); ++i) {
        if (vortices.circulation[i] == 0.0)
            continue;
        for (std::size_t j = i + 1; j < vortices.size(); ++j) {
            if (vortices.circulation[j] == 0.0 ||
                std::signbit(vortices.circulation[i]) == std::signbit(vortices.circulation[j]))
                continue;
            const double dx =
                displacement(vortices.x[i] - vortices.x[j], params_.boxLengthX, periodicX);
            const double dy =
                displacement(vortices.y[i] - vortices.y[j], params_.boxLengthY, periodicY);
            const double distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < thresholdSquared)
                candidates.push_back({i, j, distanceSquared});
        }
        if (params_.boundaryCondition == "disk") {
            const double radiusSquared = params_.diskRadius * params_.diskRadius;
            const double radialSquared =
                vortices.x[i] * vortices.x[i] + vortices.y[i] * vortices.y[i];
            // The circle-theorem image has radius R^2/r. The removal parameter
            // is the full real/image dipole separation, R^2/r-r. It is not
            // exactly twice the geometric wall gap R-r for a curved wall.
            if (radialSquared > 0.0) {
                const double imageDistance =
                    (radiusSquared - radialSquared) / std::sqrt(radialSquared);
                if (imageDistance * imageDistance < thresholdSquared)
                    candidates.push_back({i, vortices.size(), imageDistance * imageDistance});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &left, const Candidate &right) {
                  if (left.distanceSquared != right.distanceSquared)
                      return left.distanceSquared < right.distanceSquared;
                  return std::pair{left.first, left.second} < std::pair{right.first, right.second};
              });

    std::vector<bool> selected(vortices.size(), false);
    std::vector<RemovedEvent> events;
    for (const Candidate &candidate : candidates) {
        if (selected[candidate.first] ||
            (candidate.second < vortices.size() && selected[candidate.second]))
            continue;
        selected[candidate.first] = true;
        if (candidate.second < vortices.size()) {
            selected[candidate.second] = true;
            events.push_back({vortices.circulation[candidate.first],
                              vortices.circulation[candidate.second], false});
        } else {
            events.push_back({vortices.circulation[candidate.first], 0.0, true});
        }
    }
    if (events.empty())
        return 0;

    const std::size_t originalPopulation = vortices.size();
    VortexSystem survivors;
    const std::size_t selectedCount =
        static_cast<std::size_t>(std::count(selected.begin(), selected.end(), true));
    survivors.reserve(originalPopulation - selectedCount);
    for (std::size_t i = 0; i < originalPopulation; ++i) {
        if (!selected[i])
            survivors.pushBack(vortices.x[i], vortices.y[i], vortices.circulation[i]);
    }
    vortices = std::move(survivors);
    removedPairs_ += events.size();

    if (params_.dipoleReinjection != ReinjectionMode::none) {
        for (const RemovedEvent &event : events) {
            if (event.wallImage)
                injectSingle(vortices, event.firstCirculation);
            else
                injectPair(vortices, event.firstCirculation, event.secondCirculation,
                           originalPopulation);
        }
        reinjectedPairs_ += events.size();
    }
    return events.size();
}

void DipoleManager::injectSingle(VortexSystem &vortices, double circulation) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double radius = params_.diskRadius * std::sqrt(unit(random_));
    const double angle = 2.0 * std::numbers::pi * unit(random_);
    vortices.pushBack(radius * std::cos(angle), radius * std::sin(angle), circulation);
}

void DipoleManager::injectPair(VortexSystem &vortices, double firstCirculation,
                               double secondCirculation, std::size_t population) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const bool periodic = params_.boundaryCondition == "periodic";
    const double area = periodic ? params_.boxLengthX * params_.boxLengthY
                                 : std::numbers::pi * params_.diskRadius * params_.diskRadius;
    const double spacing = std::sqrt(area / static_cast<double>(population));

    const auto randomPosition = [&]() {
        if (periodic)
            return std::pair{(unit(random_) - 0.5) * params_.boxLengthX,
                             (unit(random_) - 0.5) * params_.boxLengthY};
        const double radius = params_.diskRadius * std::sqrt(unit(random_));
        const double angle = 2.0 * std::numbers::pi * unit(random_);
        return std::pair{radius * std::cos(angle), radius * std::sin(angle)};
    };

    auto first = randomPosition();
    auto second = randomPosition();
    if (params_.dipoleReinjection == ReinjectionMode::paired) {
        constexpr std::size_t maximumAttempts = 100000;
        bool placed = false;
        for (std::size_t attempt = 0; attempt < maximumAttempts; ++attempt) {
            first = randomPosition();
            const double angle = 2.0 * std::numbers::pi * unit(random_);
            second = {first.first + spacing * std::cos(angle),
                      first.second + spacing * std::sin(angle)};
            if (periodic) {
                second.first = std::remainder(second.first, params_.boxLengthX);
                second.second = std::remainder(second.second, params_.boxLengthY);
                placed = true;
                break;
            }
            if (second.first * second.first + second.second * second.second <
                params_.diskRadius * params_.diskRadius) {
                placed = true;
                break;
            }
        }
        if (!placed)
            throw std::runtime_error("could not find a valid paired reinjection position");
    }
    vortices.pushBack(first.first, first.second, firstCirculation);
    vortices.pushBack(second.first, second.second, secondCirculation);
}

DipoleEventState DipoleManager::state() const {
    std::ostringstream output;
    output << random_;
    return {removedPairs_, reinjectedPairs_, output.str()};
}
