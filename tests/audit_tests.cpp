#include "compute.h"
#include "initial_condition.h"
#include "read.h"
#include "timestep.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
void near(double actual, double expected, double tolerance, const char *label) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance)
        throw std::runtime_error(std::string(label) + " failed");
}
template <typename Function> void rejects(Function function, const char *label) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was not rejected");
}

void closeEncounterTests() {
    constexpr double inverseTwoPi = 0.5 / std::numbers::pi;
    VortexSystem state(2);
    state.x = {0, 1e-10};
    state.circulation = {1, -1};
    VelocityField velocity;
    PeriodicBoxKernel box(2, 2);
    box.evaluate(state, velocity);
    near(velocity.y[0] / (inverseTwoPi / 1e-10), 1, 1e-12, "close periodic pair");
    if (!std::isfinite(box.hamiltonian(state)))
        throw std::runtime_error("close periodic energy is non-finite");
    state.x[1] = 0;
    rejects([&] { box.evaluate(state, velocity); }, "coincident periodic pair");
    state.x = {0, 1e-10};
    PeriodicXKernel cylinder(2);
    cylinder.evaluate(state, velocity);
    near(velocity.y[0] / (inverseTwoPi / 1e-10), 1, 1e-12, "close singly periodic pair");
    if (!std::isfinite(cylinder.hamiltonian(state)))
        throw std::runtime_error("close singly periodic energy is non-finite");
    state.x[1] = 0;
    rejects([&] { cylinder.evaluate(state, velocity); }, "coincident singly periodic pair");
    state.x = {0.2, 1e-155};
    state.circulation = {1, 1};
    DiskKernel disk(1);
    disk.evaluate(state, velocity);
    near(velocity.x[0], 0, 1e-15, "near-center disk radial velocity");
    near(velocity.y[0], inverseTwoPi * (5 + .2 / .96), 1e-14,
         "near-center disk tangential velocity");
    near(velocity.y[1], inverseTwoPi * (-5 + .2), 1e-14, "near-center source image");
    rejects([] { PeriodicBoxKernel invalid(2, 2, 65); }, "excess periodic layers");
}

void hamiltonianTests() {
    VortexSystem state(4);
    state.x = {-.31, .23, .11, -.14};
    state.y = {-.12, .14, -.29, .32};
    state.circulation = {1, -1, 2, -2};
    InfinitePlaneKernel plane(.05);
    PeriodicXKernel cylinder(2);
    PeriodicBoxKernel box(2, 2);
    DiskKernel disk(1);
    for (const VelocityKernel *kernel :
         {static_cast<const VelocityKernel *>(&plane),
          static_cast<const VelocityKernel *>(&cylinder), static_cast<const VelocityKernel *>(&box),
          static_cast<const VelocityKernel *>(&disk)}) {
        VelocityField velocity;
        kernel->evaluate(state, velocity);
        constexpr double h = 1e-6;
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto plus = state, minus = state;
            plus.x[i] += h;
            minus.x[i] -= h;
            const double dx = (kernel->hamiltonian(plus) - kernel->hamiltonian(minus)) / (2 * h);
            plus = state;
            minus = state;
            plus.y[i] += h;
            minus.y[i] -= h;
            const double dy = (kernel->hamiltonian(plus) - kernel->hamiltonian(minus)) / (2 * h);
            near(velocity.x[i], dy / state.circulation[i], 2e-8, "Hamiltonian u derivative");
            near(velocity.y[i], -dx / state.circulation[i], 2e-8, "Hamiltonian v derivative");
        }
        auto evolved = state;
        RungeKuttaIntegrator integrator(state.size());
        for (int step = 0; step < 100; ++step)
            integrator.rk4Step(evolved, 1e-4, *kernel);
        near(kernel->hamiltonian(evolved), kernel->hamiltonian(state), 1e-10,
             "multi-vortex Hamiltonian conservation");
    }
}

class NonFiniteKernel : public VelocityKernel {
  public:
    void evaluateRange(const std::vector<double> &x, const std::vector<double> &,
                       const std::vector<double> &, VelocityField &v, std::size_t,
                       std::size_t) const override {
        v.resize(x.size());
        for (double &value : v.x)
            value = std::numeric_limits<double>::quiet_NaN();
    }
    double hamiltonian(const VortexSystem &) const override { return 0; }
};

void invalidNumericTests() {
    VortexSystem state(1);
    state.circulation[0] = 1;
    NonFiniteKernel bad;
    RungeKuttaIntegrator rk(1);
    SimParams params;
    rejects([&] { rk.dopri5Step(state, .01, bad, params); }, "NaN adaptive derivative");
    rejects([&] { rk.rk4Step(state, .01, bad); }, "NaN RK4 derivative");
    InfinitePlaneKernel plane;
    rejects([&] { rk.rk4Step(state, 0, plane); }, "zero timestep");
    params.endTime = std::numeric_limits<double>::infinity();
    rejects([&] { params.validate(); }, "infinite end time");
    params = SimParams{};
    params.coreRadius = 1e200;
    rejects([&] { params.validate(); }, "overflowed core radius square");
    params = SimParams{};
    params.boundaryCondition = "periodic_x";
    params.validate();
    params.dipoleRemoval = true;
    params.dipoleReinjection = ReinjectionMode::paired;
    rejects([&] { params.validate(); }, "singly periodic reinjection");
    state.x[0] = std::numeric_limits<double>::quiet_NaN();
    VelocityField velocity;
    rejects([&] { plane.evaluate(state, velocity); }, "non-finite initial position");
    rejects([&] { computeInvariants(state, plane); }, "non-finite invariant position");
}

void generatorValidationTests() {
    InitialConditionOptions options;
    options.count = 2;
    options.minimumSeparation = -1;
    rejects([&] { generateInitialCondition(options); }, "negative generator separation");
    options.minimumSeparation = 0;
    options.circulationMagnitude = std::numeric_limits<double>::infinity();
    rejects([&] { generateInitialCondition(options); }, "infinite generator circulation");
    options.circulationMagnitude = 1;
    options.geometry = InitialGeometry::periodic;
    options.count = 3;
    rejects([&] { generateInitialCondition(options); }, "odd periodic population");
}
} // namespace

int main() {
    try {
        closeEncounterTests();
        hamiltonianTests();
        invalidNumericTests();
        generatorValidationTests();
        std::cout << "numerical audit tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << "audit test failure: " << error.what() << '\n';
        return 1;
    }
}
