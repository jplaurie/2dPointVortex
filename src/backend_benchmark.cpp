#include "backend.h"
#include "timestep.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

std::size_t positiveInteger(const char *text, const char *name) {
    const std::string token(text);
    if (token.empty() || token.front() == '-')
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    std::size_t parsed = 0;
    const auto value = std::stoull(token, &parsed);
    if (parsed != token.size() || value == 0 ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return static_cast<std::size_t>(value);
}

VortexSystem makeBenchmarkState(std::size_t count) {
    VortexSystem state(count);
    // A deterministic ring avoids initialization noise and coincident particles.  A small
    // regularization radius and timestep keep the state well behaved for long calibrated runs.
    for (std::size_t i = 0; i < count; ++i) {
        const double angle =
            2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(count);
        state.x[i] = std::cos(angle);
        state.y[i] = std::sin(angle);
        state.circulation[i] = i % 2 == 0 ? 1.0 : -1.0;
    }
    return state;
}

void advance(std::size_t steps, VortexSystem &state, RungeKuttaIntegrator &integrator,
             const VelocityKernel &kernel, DeviceStepper *deviceStepper) {
    constexpr double timeStep = 1.0e-6;
    for (std::size_t step = 0; step < steps; ++step) {
        if (deviceStepper)
            deviceStepper->rk4Step(timeStep);
        else
            integrator.rk4Step(state, timeStep, kernel);
    }
}

double checksum(const VortexSystem &state) {
    double value = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i)
        value += static_cast<double>(i + 1) * (state.x[i] + 0.5 * state.y[i]);
    return value;
}

} // namespace

int main(int argc, char **argv) {
    bool initialized = false;
    try {
        backendInitialize(argc, argv);
        initialized = true;
        if (argc < 3 || argc > 4)
            throw std::invalid_argument("usage: backend_benchmark N timed_steps [warmup_steps]");

        const std::size_t count = positiveInteger(argv[1], "N");
        const std::size_t timedSteps = positiveInteger(argv[2], "timed_steps");
        const std::size_t warmupSteps = argc == 4 ? positiveInteger(argv[3], "warmup_steps") : 2;

        SimParams params;
        params.boundary = BoundaryKind::infinite;
        params.integrator = IntegratorKind::rk4;
        params.coreRadius = 1.0e-3;

        VortexSystem state = makeBenchmarkState(count);
        auto kernel = makeBackendKernel(params);
        auto *deviceStepper = dynamic_cast<DeviceStepper *>(kernel.get());
        RungeKuttaIntegrator integrator(count);
        if (deviceStepper)
            deviceStepper->uploadState(state);

        // Backend construction, allocations, CUDA context creation/state upload, and this warmup
        // are deliberately outside the measured interval.
        advance(warmupSteps, state, integrator, *kernel, deviceStepper);

        const auto start = std::chrono::steady_clock::now();
        advance(timedSteps, state, integrator, *kernel, deviceStepper);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        // Device-to-host transfer and checksum are also outside the measured interval.
        if (deviceStepper)
            deviceStepper->downloadState(state);
        if (backendIsRoot()) {
            const double interactions = 4.0 * static_cast<double>(timedSteps) *
                                        static_cast<double>(count) * static_cast<double>(count - 1);
            std::cout << std::setprecision(17) << "backend=" << backendName() << " N=" << count
                      << " steps=" << timedSteps << " warmup_steps=" << warmupSteps
                      << " seconds=" << seconds
                      << " seconds_per_step=" << seconds / static_cast<double>(timedSteps)
                      << " interactions_per_second=" << interactions / seconds
                      << " checksum=" << checksum(state) << '\n';
        }
    } catch (const std::exception &error) {
        if (backendIsRoot())
            std::cerr << "error: " << error.what() << '\n';
        if (initialized)
            backendAbort(1);
        return 1;
    }
    if (initialized)
        backendFinalize();
    return 0;
}
