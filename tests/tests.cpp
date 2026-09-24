#include "checkpoint.h"
#include "compute.h"
#include "dipole.h"
#include "initial_condition.h"
#include "read.h"
#include "timestep.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <stdexcept>
namespace {
void near(double got, double want, double tol, const char *what) {
    if (!std::isfinite(got) || !std::isfinite(want) || std::abs(got - want) > tol)
        throw std::runtime_error(std::string(what) + " failed");
}
VortexSystem pair() {
    VortexSystem s(2);
    s.x = {-1.0, 1.0};
    s.y = {0.0, 0.0};
    s.circulation = {1.0, 1.0};
    return s;
}
void velocityTests() {
    VortexSystem one(1);
    one.x[0] = 2;
    one.y[0] = -3;
    one.circulation[0] = 4;
    VelocityField velocity;
    InfinitePlaneKernel kernel;
    kernel.evaluate(one, velocity);
    near(velocity.x[0], 0, 0, "single u");
    near(velocity.y[0], 0, 0, "single v");
    const auto two = pair();
    kernel.evaluate(two, velocity);
    const double speed = 1.0 / (4.0 * std::numbers::pi);
    near(velocity.y[0], -speed, 1e-15, "pair v0");
    near(velocity.y[1], speed, 1e-15, "pair v1");
}
void rk4Test() {
    auto s = pair();
    InfinitePlaneKernel kernel;
    RungeKuttaIntegrator rk(s.size());
    for (int i = 0; i < 1000; ++i)
        rk.rk4Step(s, 1e-3, kernel);
    const double angle = 1.0 / (4.0 * std::numbers::pi);
    near(s.x[1], std::cos(angle), 2e-12, "RK4 x");
    near(s.y[1], std::sin(angle), 2e-12, "RK4 y");
}
void dopriTest() {
    auto s = pair();
    InfinitePlaneKernel kernel;
    RungeKuttaIntegrator rk(s.size());
    SimParams p;
    p.absoluteTolerance = 1e-12;
    p.relativeTolerance = 1e-10;
    p.minimumTimeStep = 1e-12;
    p.maximumTimeStep = 0.2;
    double time = 0, dt = 0.01;
    while (time < 1.0) {
        auto r = rk.dopri5Step(s, std::min(dt, 1.0 - time), kernel, p);
        time += r.acceptedTimeStep;
        dt = r.suggestedTimeStep;
    }
    const double angle = 1.0 / (4.0 * std::numbers::pi);
    near(s.x[1], std::cos(angle), 2e-10, "DOPRI x");
    near(s.y[1], std::sin(angle), 2e-10, "DOPRI y");
}
void managedRunPathTest() {
    const SimParams defaults;
    const RunPaths paths = runPaths(defaults);
    if (defaults.runDirectory != "runs/default" ||
        paths.trajectory != "runs/default/trajectory.csv" ||
        paths.diagnostics != "runs/default/diagnostics.csv" ||
        paths.checkpoints != "runs/default/checkpoints")
        throw std::runtime_error("managed run layout defaults failed");
}
void checkpointTest() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("point_vortex_checkpoint_test_" +
         std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    const auto state = pair();
    const auto initial = computeInvariants(state);
    SimParams parameters;
    const DipoleEventState dipoleState = DipoleManager(parameters).state();
    const OutputSchedule schedule{0.1, 0.2, 0.5, 1.4, 1.5};
    const CheckpointProgress progress{1.25, 0.0125, 1.3, 42, 7, 9};
    writeCheckpoint(directory, state, initial, parameters, initial, dipoleState, schedule,
                    progress);
    const auto restored = loadCheckpoint(checkpointPath(directory, 7));
    if (!restored.hasOutputSchedule)
        throw std::runtime_error("checkpoint output schedule missing");
    near(restored.outputSchedule.nextDiagnosticsTime, 1.4, 0.0, "next diagnostics time");
    near(restored.outputSchedule.nextCheckpointTime, 1.5, 0.0, "next checkpoint time");
    near(restored.outputSchedule.trajectoryInterval, 0.1, 0.0, "trajectory interval");
    near(restored.outputSchedule.diagnosticsInterval, 0.2, 0.0, "diagnostics interval");
    near(restored.outputSchedule.checkpointInterval, 0.5, 0.0, "checkpoint interval");
    near(restored.time, 1.25, 0.0, "checkpoint time");
    near(restored.suggestedTimeStep, 0.0125, 0.0, "checkpoint timestep");
    near(restored.vortices.x[1], state.x[1], 0.0, "checkpoint position");
    near(restored.initialInvariants.hamiltonian, initial.hamiltonian, 0.0, "checkpoint invariant");
    near(restored.segmentInvariants.hamiltonian, initial.hamiltonian, 0.0,
         "checkpoint segment invariant");
    if (restored.acceptedSteps != 42 || restored.outputIndex != 7 || restored.eventIndex != 9)
        throw std::runtime_error("checkpoint counters failed");
    if (restored.boundaryCondition != "infinite" || restored.periodicImageLayers != 0)
        throw std::runtime_error("checkpoint geometry failed");
    if (restored.dipoleRemoval || restored.dipoleReinjection != ReinjectionMode::none ||
        restored.dipoleState.randomEngineState != dipoleState.randomEngineState)
        throw std::runtime_error("checkpoint dipole state failed");
    bool refusedOverwrite = false;
    try {
        writeCheckpoint(directory, state, initial, parameters, initial, dipoleState, schedule,
                        progress);
    } catch (const std::runtime_error &) {
        refusedOverwrite = true;
    }
    if (!refusedOverwrite)
        throw std::runtime_error("checkpoint overwrite was not refused");
    writeCheckpoint(directory, state, initial, parameters, initial, dipoleState, schedule, progress,
                    true);
    parameters.boundaryCondition = "periodic_x";
    parameters.boxLengthX = 3.0;
    CheckpointProgress periodicProgress = progress;
    periodicProgress.outputIndex = 8;
    writeCheckpoint(directory, state, initial, parameters, initial, dipoleState, schedule,
                    periodicProgress);
    const auto periodicRestored = loadCheckpoint(checkpointPath(directory, 8));
    if (periodicRestored.boundaryCondition != "periodic_x" ||
        periodicRestored.geometryLengthX != 3.0 || periodicRestored.geometryLengthY != 0.0 ||
        periodicRestored.periodicImageLayers != 0)
        throw std::runtime_error("singly periodic checkpoint geometry failed");
    std::filesystem::remove_all(directory);
}
void geometryTests() {
    VortexSystem periodic(2);
    periodic.x = {-0.4, 0.4};
    periodic.y = {0.1, -0.1};
    periodic.circulation = {1.0, -1.0};
    PeriodicBoxKernel box(2.0, 2.0, 10);
    VelocityField first, shifted;
    box.evaluate(periodic, first);
    periodic.x[0] += 2.0;
    periodic.y[1] -= 2.0;
    box.evaluate(periodic, shifted);
    for (std::size_t i = 0; i < 2; ++i) {
        near(shifted.x[i], first.x[i], 2e-14, "periodic u invariance");
        near(shifted.y[i], first.y[i], 2e-14, "periodic v invariance");
    }
    VortexSystem periodicX(2);
    periodicX.x = {0.0, 0.5};
    periodicX.y = {0.0, 0.0};
    periodicX.circulation = {1.0, 1.0};
    PeriodicXKernel cylinder(2.0);
    cylinder.evaluate(periodicX, first);
    near(first.x[0], 0.0, 0.0, "singly periodic pair u");
    near(first.y[0], -0.25, 2e-16, "singly periodic pair v");
    periodicX.x[1] += 2.0;
    cylinder.evaluate(periodicX, shifted);
    for (std::size_t i = 0; i < 2; ++i) {
        near(shifted.x[i], first.x[i], 2e-16, "singly periodic u invariance");
        near(shifted.y[i], first.y[i], 2e-16, "singly periodic v invariance");
    }
    periodicX.x = {0.0, 0.0};
    periodicX.y = {-100.0, 100.0};
    cylinder.evaluate(periodicX, first);
    near(first.x[0], 0.25, 1e-15, "singly periodic lower far field");
    near(first.x[1], -0.25, 1e-15, "singly periodic upper far field");
    near(first.y[0], 0.0, 0.0, "singly periodic far-field transverse velocity");
    VortexSystem disk(1);
    disk.x[0] = 0.5;
    disk.y[0] = 0.0;
    disk.circulation[0] = 1.0;
    DiskKernel circle(2.0);
    circle.evaluate(disk, first);
    near(first.x[0], 0.0, 1e-15, "disk radial velocity");
    near(first.y[0], 0.5 / (2.0 * std::numbers::pi * (4.0 - 0.25)), 1e-15, "disk image velocity");
    const double diskEnergy = circle.hamiltonian(disk);
    RungeKuttaIntegrator diskIntegrator(1);
    for (int step = 0; step < 100; ++step)
        diskIntegrator.rk4Step(disk, 1e-3, circle);
    near(circle.hamiltonian(disk), diskEnergy, 2e-15, "disk Hamiltonian");
}
void periodicInitializationTest() {
    VortexSystem first(100), second(100);
    initializePeriodicVortices(first, 2.0, 4.0, 42);
    initializePeriodicVortices(second, 2.0, 4.0, 42);
    double totalCirculation = 0.0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (first.x[i] < -1.0 || first.x[i] > 1.0 || first.y[i] < -2.0 || first.y[i] > 2.0)
            throw std::runtime_error("periodic initial position outside box");
        near(first.x[i], second.x[i], 0.0, "periodic initialization reproducibility");
        near(first.y[i], second.y[i], 0.0, "periodic initialization reproducibility");
        totalCirculation += first.circulation[i];
    }
    near(totalCirculation, 0.0, 0.0, "periodic initial circulation");
}
void dipoleRemovalTest() {
    SimParams parameters;
    parameters.dipoleRemoval = true;
    parameters.dipoleRemovalDistance = 0.05;
    VortexSystem state(3);
    state.x = {0.0, 0.01, 0.02};
    state.y = {0.0, 0.0, 0.0};
    state.circulation = {1.0, -1.0, -1.0};
    DipoleManager manager(parameters);
    if (manager.process(state) != 1 || state.size() != 1 || state.x[0] != 0.02)
        throw std::runtime_error("closest-first dipole removal failed");
    const auto events = manager.state();
    if (events.removedPairs != 1 || events.reinjectedPairs != 0)
        throw std::runtime_error("dipole removal counters failed");
}
void periodicReinjectionTest() {
    SimParams parameters;
    parameters.boundaryCondition = "periodic";
    parameters.boxLengthX = parameters.boxLengthY = 2.0;
    parameters.dipoleRemoval = true;
    parameters.dipoleRemovalDistance = 0.05;
    parameters.dipoleReinjection = ReinjectionMode::paired;
    parameters.randomSeed = 42;
    VortexSystem state(4);
    state.x = {0.99, -0.99, -0.5, 0.5};
    state.y = {0.0, 0.0, 0.5, -0.5};
    state.circulation = {1.0, -1.0, 1.0, -1.0};
    DipoleManager manager(parameters);
    if (manager.process(state) != 1 || state.size() != 4)
        throw std::runtime_error("periodic minimum-image dipole removal failed");
    for (std::size_t i = 0; i < state.size(); ++i)
        if (state.x[i] < -1.0 || state.x[i] > 1.0 || state.y[i] < -1.0 || state.y[i] > 1.0)
            throw std::runtime_error("periodic reinjection outside box");
    const double dx = std::remainder(state.x[2] - state.x[3], 2.0);
    const double dy = std::remainder(state.y[2] - state.y[3], 2.0);
    near(dx * dx + dy * dy, 1.0, 2e-15, "paired reinjection spacing");
    const auto saved = manager.state();
    DipoleManager restored(parameters, saved);
    if (restored.state().randomEngineState != saved.randomEngineState)
        throw std::runtime_error("reinjection random state restoration failed");
}
void periodicXDipoleRemovalTest() {
    SimParams parameters;
    parameters.boundaryCondition = "periodic_x";
    parameters.boxLengthX = 2.0;
    parameters.dipoleRemoval = true;
    parameters.dipoleRemovalDistance = 0.05;
    VortexSystem state(4);
    state.x = {0.99, -0.99, 0.99, -0.99};
    state.y = {0.0, 0.0, 0.0, 2.0};
    state.circulation = {1.0, -1.0, 1.0, -1.0};
    DipoleManager manager(parameters);
    if (manager.process(state) != 1 || state.size() != 2)
        throw std::runtime_error("singly periodic minimum-image dipole removal failed");
    near(state.y[0], 0.0, 0.0, "singly periodic surviving first y");
    near(state.y[1], 2.0, 0.0, "singly periodic unbounded y distance");
}
void diskReinjectionTest() {
    SimParams parameters;
    parameters.boundaryCondition = "disk";
    parameters.diskRadius = 1.0;
    parameters.dipoleRemoval = true;
    parameters.dipoleRemovalDistance = 0.1;
    parameters.dipoleReinjection = ReinjectionMode::independent;
    VortexSystem state(2);
    state.x = {0.0, 0.01};
    state.y = {0.0, 0.0};
    state.circulation = {2.0, -0.5};
    DipoleManager manager(parameters);
    manager.process(state);
    if (state.size() != 2 || state.circulation[0] != 2.0 || state.circulation[1] != -0.5)
        throw std::runtime_error("reinjection did not preserve circulations");
    for (std::size_t i = 0; i < state.size(); ++i)
        if (state.x[i] * state.x[i] + state.y[i] * state.y[i] >= 1.0)
            throw std::runtime_error("disk reinjection outside disk");

    parameters.dipoleReinjection = ReinjectionMode::paired;
    state.x = {0.0, 0.01};
    state.y = {0.0, 0.0};
    state.circulation = {1.0, -1.0};
    DipoleManager pairedManager(parameters);
    pairedManager.process(state);
    const double dx = state.x[0] - state.x[1];
    const double dy = state.y[0] - state.y[1];
    near(dx * dx + dy * dy, std::numbers::pi / 2.0, 3e-15, "paired disk reinjection spacing");
    for (std::size_t i = 0; i < state.size(); ++i)
        if (state.x[i] * state.x[i] + state.y[i] * state.y[i] >= 1.0)
            throw std::runtime_error("paired disk reinjection outside disk");
}
void diskWallDipoleRemovalTest() {
    SimParams parameters;
    parameters.boundaryCondition = "disk";
    parameters.diskRadius = 1.0;
    parameters.dipoleRemoval = true;
    parameters.dipoleRemovalDistance = 0.05;
    parameters.dipoleReinjection = ReinjectionMode::none;
    VortexSystem state(2);
    state.x = {0.98, 0.0};
    state.y = {0.0, 0.0};
    state.circulation = {1.0, -1.0};
    DipoleManager manager(parameters);
    if (manager.process(state) != 1 || state.size() != 1 || state.circulation[0] != -1.0)
        throw std::runtime_error("disk wall-image dipole removal failed");
    const auto events = manager.state();
    if (events.removedPairs != 1 || events.reinjectedPairs != 0)
        throw std::runtime_error("disk wall-image event counters failed");

    // For a circular wall the exact image separation is R^2/r-r. The
    // geometric wall gap R-r is only asymptotically half of that distance.
    const double radius = 0.98;
    const double imageRadius = 1.0 / radius;
    const double imageDistance = imageRadius - radius;
    const double wallGap = 1.0 - radius;
    near(imageDistance, (1.0 - radius * radius) / radius, 1e-15, "disk wall-image distance");
    if (std::abs(2.0 * wallGap - imageDistance) < 1e-6)
        throw std::runtime_error(
            "curved-wall image distance was treated as exactly twice the wall gap");

    parameters.dipoleReinjection = ReinjectionMode::paired;
    state = VortexSystem(1);
    state.x[0] = 0.98;
    state.circulation[0] = 2.0;
    DipoleManager reinjected(parameters);
    if (reinjected.process(state) != 1 || state.size() != 1 || state.circulation[0] != 2.0)
        throw std::runtime_error("disk wall-image reinjection did not preserve population");
    if (state.x[0] * state.x[0] + state.y[0] * state.y[0] >= 1.0)
        throw std::runtime_error("disk wall-image reinjection outside disk");
}
class CountingKernel final : public VelocityKernel {
  public:
    mutable std::size_t evaluations = 0;
    void evaluateRange(const std::vector<double> &x, const std::vector<double> &y,
                       const std::vector<double> &gamma, VelocityField &velocity, std::size_t begin,
                       std::size_t end) const override {
        ++evaluations;
        kernel.evaluateRange(x, y, gamma, velocity, begin, end);
    }
    double hamiltonian(const VortexSystem &state) const override {
        return kernel.hamiltonian(state);
    }

  private:
    InfinitePlaneKernel kernel;
};
void fsalTest() {
    auto state = pair();
    CountingKernel kernel;
    RungeKuttaIntegrator integrator(state.size());
    SimParams p;
    p.absoluteTolerance = 1e-10;
    p.relativeTolerance = 1e-8;
    integrator.dopri5Step(state, 1e-3, kernel, p);
    if (kernel.evaluations != 7)
        throw std::runtime_error("first DOPRI step did not use 7 stages");
    integrator.dopri5Step(state, 1e-3, kernel, p);
    if (kernel.evaluations != 13)
        throw std::runtime_error("DOPRI FSAL stage was not reused");
    integrator.invalidateCachedDerivative();
    integrator.dopri5Step(state, 1e-3, kernel, p);
    if (kernel.evaluations != 20)
        throw std::runtime_error("DOPRI FSAL cache was not invalidated after an event");
}
void initialConditionGeneratorTest() {
    InitialConditionOptions periodic;
    periodic.geometry = InitialGeometry::periodic;
    periodic.pattern = InitialPattern::random;
    periodic.count = 100;
    periodic.seed = 987654;
    periodic.boxLength = 2.0;
    periodic.minimumSeparation = 0.02;
    const VortexSystem first = generateInitialCondition(periodic);
    const VortexSystem second = generateInitialCondition(periodic);
    if (first.x != second.x || first.y != second.y || first.circulation != second.circulation)
        throw std::runtime_error("initial-condition generator is not reproducible");
    if (minimumVortexSeparation(first, periodic) < periodic.minimumSeparation)
        throw std::runtime_error("generated periodic separation constraint failed");
    double circulation = 0.0;
    for (double value : first.circulation)
        circulation += value;
    near(circulation, 0.0, 0.0, "generated periodic neutrality");

    InitialConditionOptions disk;
    disk.geometry = InitialGeometry::disk;
    disk.pattern = InitialPattern::random;
    disk.count = 50;
    disk.seed = 1234;
    disk.diskRadius = 2.0;
    disk.minimumSeparation = 0.03;
    const VortexSystem diskState = generateInitialCondition(disk);
    validateInitialCondition(diskState, disk);
    for (std::size_t i = 0; i < diskState.size(); ++i)
        if (diskState.x[i] * diskState.x[i] + diskState.y[i] * diskState.y[i] >= 4.0)
            throw std::runtime_error("generated disk vortex lies outside disk");

    InitialConditionOptions periodicX;
    periodicX.geometry = InitialGeometry::periodic_x;
    periodicX.pattern = InitialPattern::single;
    const VortexSystem periodicXState = generateInitialCondition(periodicX);
    if (periodicXState.size() != 1 || periodicXState.circulation[0] != 1.0)
        throw std::runtime_error("singly periodic non-neutral initial condition failed");

    InitialConditionOptions dipole;
    dipole.geometry = InitialGeometry::periodic;
    dipole.pattern = InitialPattern::dipole;
    const VortexSystem dipoleState = generateInitialCondition(dipole);
    if (dipoleState.size() != 2 || dipoleState.circulation[0] != 1.0 ||
        dipoleState.circulation[1] != -1.0)
        throw std::runtime_error("generated dipole case failed");

    const auto filename =
        std::filesystem::temp_directory_path() /
        ("point_vortex_initial_test_" +
         std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
         ".dat");
    writeInitialCondition(filename.string(), first, periodic);
    const InitialConditionMetadata metadata = readInitialConditionMetadata(filename.string());
    if (!metadata.geometry || *metadata.geometry != "periodic" || !metadata.boxLength ||
        *metadata.boxLength != periodic.boxLength)
        throw std::runtime_error("generated initial-condition metadata failed");
    const VortexSystem loaded = loadVortices(filename.string());
    std::filesystem::remove(filename);
    if (loaded.x != first.x || loaded.y != first.y || loaded.circulation != first.circulation)
        throw std::runtime_error("generated initial-condition round trip failed");

    InitialConditionOptions invalidPair;
    invalidPair.geometry = InitialGeometry::periodic;
    invalidPair.pattern = InitialPattern::corotatingPair;
    bool rejected = false;
    try {
        generateInitialCondition(invalidPair);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("non-neutral periodic test case was not rejected");
}
} // namespace
int main() {
    try {
        velocityTests();
        rk4Test();
        dopriTest();
        managedRunPathTest();
        checkpointTest();
        geometryTests();
        periodicInitializationTest();
        dipoleRemovalTest();
        periodicReinjectionTest();
        periodicXDipoleRemovalTest();
        diskReinjectionTest();
        diskWallDipoleRemovalTest();
        fsalTest();
        initialConditionGeneratorTest();
        std::cout << "all point-vortex tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << "test failure: " << e.what() << '\n';
        return 1;
    }
}
