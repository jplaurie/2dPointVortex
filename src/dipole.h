#ifndef POINT_VORTEX_DIPOLE_H
#define POINT_VORTEX_DIPOLE_H

#include "params.h"
#include "vortex.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

struct DipoleEventState {
    std::size_t removedPairs = 0;
    std::size_t reinjectedPairs = 0;
    std::string randomEngineState;
    std::size_t removedUpperPairs = 0;
};

class DipoleManager {
  public:
    explicit DipoleManager(const SimParams &params);
    DipoleManager(const SimParams &params, const DipoleEventState &state);

    // Build a closest-first, disjoint matching of real opposite-sign vortices and
    // remove matches below the lower threshold or, when enabled, above the upper
    // threshold. In a disk, a real vortex and its opposite-sign circle-theorem
    // image are also a lower-threshold candidate; that wall event removes only
    // the real vortex.
    // Returns the number of pairs processed during this call.
    std::size_t process(VortexSystem &vortices);
    [[nodiscard]] DipoleEventState state() const;

  private:
    const SimParams &params_;
    std::mt19937_64 random_;
    std::size_t removedPairs_ = 0;
    std::size_t removedUpperPairs_ = 0;
    std::size_t reinjectedPairs_ = 0;

    void injectPair(VortexSystem &vortices, double firstCirculation, double secondCirculation,
                    std::size_t population);
    void injectSingle(VortexSystem &vortices, double circulation);
};

#endif
