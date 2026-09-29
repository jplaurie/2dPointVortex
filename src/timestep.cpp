#include "timestep.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
RungeKuttaIntegrator::RungeKuttaIntegrator(std::size_t n) : initial_(n), temporary_(n) {
    for (auto &stage : stages_)
        stage.resize(n);
}
void RungeKuttaIntegrator::ensureSize(const VortexSystem &state) {
    state.validate();
    if (temporary_.size() != state.size()) {
        initial_.resize(state.size());
        temporary_.resize(state.size());
        for (auto &stage : stages_)
            stage.resize(state.size());
        fsalValid_ = false;
    }
}
void RungeKuttaIntegrator::makeStage(std::size_t stage,
                                     const std::array<double, 7> &coefficients, double dt) {
    for (std::size_t i = 0; i < initial_.size(); ++i) {
        double dx = 0.0, dy = 0.0;
        for (std::size_t j = 0; j < stage; ++j) {
            dx += coefficients[j] * stages_[j].x[i];
            dy += coefficients[j] * stages_[j].y[i];
        }
        temporary_.x[i] = initial_.x[i] + dt * dx;
        temporary_.y[i] = initial_.y[i] + dt * dy;
    }
}
void RungeKuttaIntegrator::rk4Step(VortexSystem &state, double dt, const VelocityKernel &kernel) {
    if (!std::isfinite(dt) || !(dt > 0.0))
        throw std::invalid_argument("timestep must be finite and positive");
    ensureSize(state);
    initial_.x = state.x;
    initial_.y = state.y;
    fsalValid_ = false;
    kernel.evaluate(initial_.x, initial_.y, state.circulation, stages_[0]);
    const std::array<double, 7> a2 = {0.5};
    makeStage(1, a2, dt);
    kernel.evaluate(temporary_.x, temporary_.y, state.circulation, stages_[1]);
    const std::array<double, 7> a3 = {0.0, 0.5};
    makeStage(2, a3, dt);
    kernel.evaluate(temporary_.x, temporary_.y, state.circulation, stages_[2]);
    const std::array<double, 7> a4 = {0.0, 0.0, 1.0};
    makeStage(3, a4, dt);
    kernel.evaluate(temporary_.x, temporary_.y, state.circulation, stages_[3]);
    for (std::size_t i = 0; i < state.size(); ++i) {
        temporary_.x[i] = initial_.x[i] + dt *
                                              (stages_[0].x[i] + 2.0 * stages_[1].x[i] +
                                               2.0 * stages_[2].x[i] + stages_[3].x[i]) /
                                              6.0;
        temporary_.y[i] = initial_.y[i] + dt *
                                              (stages_[0].y[i] + 2.0 * stages_[1].y[i] +
                                               2.0 * stages_[2].y[i] + stages_[3].y[i]) /
                                              6.0;
    }
    validateVortexArrays(temporary_.x, temporary_.y, state.circulation);
    state.x.swap(temporary_.x);
    state.y.swap(temporary_.y);
}
StepResult RungeKuttaIntegrator::dopri5Step(VortexSystem &state, double dt,
                                            const VelocityKernel &kernel,
                                            const SimParams &params) {
    if (!std::isfinite(dt) || !(dt > 0.0))
        throw std::invalid_argument("timestep must be finite and positive");
    ensureSize(state);
    initial_.x = state.x;
    initial_.y = state.y;
    static constexpr double b5[7] = {
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
    static constexpr double b4[7] = {
        5179.0 / 57600.0, 0.0,       7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0,
        187.0 / 2100.0,   1.0 / 40.0};
    unsigned rejected = 0;
    // Dormand--Prince is FSAL: k7 from an accepted step is k1 of the next step.
    if (!fsalValid_)
        kernel.evaluate(initial_.x, initial_.y, state.circulation, stages_[0]);
    for (;;) {
        for (std::size_t s = 1; s < 7; ++s) {
            makeStage(s, integrator_detail::dopriCoefficients[s], dt);
            kernel.evaluate(temporary_.x, temporary_.y, state.circulation, stages_[s]);
        }
        double error = 0.0;
        for (std::size_t i = 0; i < state.size(); ++i) {
            // Stage 7 was evaluated at the fifth-order solution. Accept exactly
            // that state so the cached FSAL derivative belongs to the saved positions.
            const double x5 = temporary_.x[i], y5 = temporary_.y[i];
            double xError = 0.0, yError = 0.0;
            for (std::size_t s = 0; s < 7; ++s) {
                xError += dt * (b5[s] - b4[s]) * stages_[s].x[i];
                yError += dt * (b5[s] - b4[s]) * stages_[s].y[i];
            }
            const double xScale =
                params.absoluteTolerance +
                params.relativeTolerance * std::max(std::abs(initial_.x[i]), std::abs(x5));
            const double yScale =
                params.absoluteTolerance +
                params.relativeTolerance * std::max(std::abs(initial_.y[i]), std::abs(y5));
            if (!std::isfinite(xError) || !std::isfinite(yError) || !std::isfinite(xScale) ||
                !std::isfinite(yScale) || !(xScale > 0.0) || !(yScale > 0.0))
                throw std::runtime_error("non-finite adaptive error estimate or invalid tolerance");
            error = std::max(error, std::abs(xError) / xScale);
            error = std::max(error, std::abs(yError) / yScale);
        }
        const double factor =
            error == 0.0 ? 5.0 : std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0);
        const double suggested =
            std::clamp(dt * factor, params.minimumTimeStep, params.maximumTimeStep);
        if (error <= 1.0) {
            state.x.swap(temporary_.x);
            state.y.swap(temporary_.y);
            stages_[0].x.swap(stages_[6].x);
            stages_[0].y.swap(stages_[6].y);
            fsalValid_ = true;
            return {dt, suggested, error, rejected};
        }
        if (dt <= params.minimumTimeStep || ++rejected > 32)
            throw std::runtime_error("adaptive integrator could not satisfy tolerance");
        dt = std::max(params.minimumTimeStep, std::min(suggested, dt * 0.9));
    }
}
