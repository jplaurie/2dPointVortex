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
    bool upperRemoval = false;
};

struct RemovalPlan {
    std::vector<bool> selected;
    std::vector<RemovedEvent> events;
};

double displacement(double difference, double length, bool periodic) {
    return periodic ? std::remainder(difference, length) : difference;
}

std::vector<Candidate> findCandidates(const VortexSystem &vortices, const SimParams &params) {
    const bool periodicX = isPeriodicX(params.boundary);
    const bool periodicY = isPeriodicY(params.boundary);
    const double lowerThresholdSquared =
        params.dipoleRemovalDistance * params.dipoleRemovalDistance;

    std::vector<Candidate> candidates;
    for (std::size_t first = 0; first < vortices.size(); ++first) {
        if (vortices.circulation[first] == 0.0)
            continue;

        for (std::size_t second = first + 1; second < vortices.size(); ++second) {
            const bool sameSign = std::signbit(vortices.circulation[first]) ==
                                  std::signbit(vortices.circulation[second]);
            if (vortices.circulation[second] == 0.0 || sameSign)
                continue;

            const double dx = displacement(vortices.x[first] - vortices.x[second],
                                           params.boxLengthX, periodicX);
            const double dy = displacement(vortices.y[first] - vortices.y[second],
                                           params.boxLengthY, periodicY);
            const double distanceSquared = dx * dx + dy * dy;

            // Upper-cutoff matching needs all separations so intermediate
            // pairs reserve each other before more distant pairs are considered.
            if (distanceSquared < lowerThresholdSquared || params.dipoleRemovalUpper)
                candidates.push_back({first, second, distanceSquared});
        }

        if (isDisk(params.boundary)) {
            const double radiusSquared = params.diskRadius * params.diskRadius;
            const double radialSquared = vortices.x[first] * vortices.x[first] +
                                         vortices.y[first] * vortices.y[first];
            // The circle-theorem image has radius R^2/r. The configured removal
            // distance is the full real/image separation, R^2/r-r, rather than
            // exactly twice the curved-wall gap R-r.
            if (radialSquared > 0.0) {
                const double imageDistance =
                    (radiusSquared - radialSquared) / std::sqrt(radialSquared);
                if (imageDistance * imageDistance < lowerThresholdSquared) {
                    candidates.push_back(
                        {first, vortices.size(), imageDistance * imageDistance});
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &left, const Candidate &right) {
                  if (left.distanceSquared != right.distanceSquared)
                      return left.distanceSquared < right.distanceSquared;
                  return std::pair{left.first, left.second} <
                         std::pair{right.first, right.second};
              });
    return candidates;
}

RemovalPlan selectRemovals(const std::vector<Candidate> &candidates,
                           const VortexSystem &vortices, const SimParams &params) {
    const double lowerThresholdSquared =
        params.dipoleRemovalDistance * params.dipoleRemovalDistance;
    const double upperThresholdSquared =
        params.dipoleRemovalUpperDistance * params.dipoleRemovalUpperDistance;

    std::vector<bool> matched(vortices.size(), false);
    RemovalPlan plan{std::vector<bool>(vortices.size(), false), {}};
    for (const Candidate &candidate : candidates) {
        const bool wallImage = candidate.second == vortices.size();
        if (matched[candidate.first] || (!wallImage && matched[candidate.second]))
            continue;

        matched[candidate.first] = true;
        if (!wallImage)
            matched[candidate.second] = true;

        const bool lowerRemoval = candidate.distanceSquared < lowerThresholdSquared;
        const bool upperRemoval = !wallImage && params.dipoleRemovalUpper &&
                                  candidate.distanceSquared > upperThresholdSquared;
        if (!lowerRemoval && !upperRemoval)
            continue;

        plan.selected[candidate.first] = true;
        if (wallImage) {
            plan.events.push_back({vortices.circulation[candidate.first], 0.0, true, false});
        } else {
            plan.selected[candidate.second] = true;
            plan.events.push_back({vortices.circulation[candidate.first],
                                   vortices.circulation[candidate.second], false, upperRemoval});
        }
    }
    return plan;
}

void keepUnselectedVortices(VortexSystem &vortices, const std::vector<bool> &selected) {
    const std::size_t selectedCount =
        static_cast<std::size_t>(std::count(selected.begin(), selected.end(), true));
    VortexSystem survivors;
    survivors.reserve(vortices.size() - selectedCount);

    for (std::size_t index = 0; index < vortices.size(); ++index) {
        if (!selected[index])
            survivors.pushBack(vortices.x[index], vortices.y[index],
                               vortices.circulation[index]);
    }
    vortices = std::move(survivors);
}
} // namespace

DipoleManager::DipoleManager(const SimParams &params)
    : params_(params), random_(params.randomSeed ^ 0xd1b54a32d192ed03ULL) {}

DipoleManager::DipoleManager(const SimParams &params, const DipoleEventState &state)
    : params_(params), random_(params.randomSeed ^ 0xd1b54a32d192ed03ULL),
      removedPairs_(state.removedPairs), removedUpperPairs_(state.removedUpperPairs),
      reinjectedPairs_(state.reinjectedPairs) {
    if (!state.randomEngineState.empty()) {
        std::istringstream input(state.randomEngineState);
        if (!(input >> random_))
            throw std::runtime_error("invalid random-generator state in checkpoint");
    }
}

std::size_t DipoleManager::process(VortexSystem &vortices) {
    if (!params_.dipoleRemoval || vortices.size() == 0)
        return 0;

    const std::vector<Candidate> candidates = findCandidates(vortices, params_);
    const RemovalPlan plan = selectRemovals(candidates, vortices, params_);
    if (plan.events.empty())
        return 0;

    const std::size_t originalPopulation = vortices.size();
    keepUnselectedVortices(vortices, plan.selected);

    removedPairs_ += plan.events.size();
    removedUpperPairs_ += static_cast<std::size_t>(
        std::count_if(plan.events.begin(), plan.events.end(),
                      [](const RemovedEvent &event) { return event.upperRemoval; }));

    if (params_.dipoleReinjection != ReinjectionMode::none) {
        for (const RemovedEvent &event : plan.events) {
            if (event.wallImage)
                injectSingle(vortices, event.firstCirculation);
            else
                injectPair(vortices, event.firstCirculation, event.secondCirculation,
                           originalPopulation);
        }
        reinjectedPairs_ += plan.events.size();
    }
    return plan.events.size();
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
    const bool periodic = isPeriodicY(params_.boundary);
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
    return {removedPairs_, reinjectedPairs_, output.str(), removedUpperPairs_};
}
