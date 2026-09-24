#include "backend.h"
#include "timestep.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cudaCheck(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

enum class Geometry : int { infinite, periodic_x, periodic, disk };

__global__ void velocityKernel(const double *x, const double *y, const double *gamma, double *u,
                               double *v, std::size_t count, std::size_t begin, std::size_t end,
                               Geometry geometry, double first, double second, int imageLayers,
                               int *failure) {
    const std::size_t target =
        begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (target >= end)
        return;
    if (!isfinite(x[target]) || !isfinite(y[target]) ||
        (geometry == Geometry::disk &&
         x[target] * x[target] + y[target] * y[target] >= first * first)) {
        atomicExch(failure, 1);
        return;
    }
    constexpr double inverseTwoPi = 0.15915494309189533576888376337251;
    constexpr double twoPi = 6.283185307179586476925286766559;
    double velocityX = 0.0, velocityY = 0.0;
    double inverseRadius = 0.0, targetX = 0.0, targetY = 0.0;
    double waveNumber = 0.0, periodicScale = 0.0;
    if (geometry == Geometry::periodic_x || geometry == Geometry::periodic) {
        waveNumber = twoPi / first;
        periodicScale = 0.5 / first;
    }
    if (geometry == Geometry::disk) {
        inverseRadius = 1.0 / first;
        targetX = x[target] * inverseRadius;
        targetY = y[target] * inverseRadius;
    }

    for (std::size_t source = 0; source < count; ++source) {
        if (geometry == Geometry::infinite) {
            if (source == target)
                continue;
            const double dx = x[target] - x[source];
            const double dy = y[target] - y[source];
            const double denominator = dx * dx + dy * dy + first;
            if (denominator == 0.0) {
                atomicExch(failure, 1);
                continue;
            }
            const double coefficient = inverseTwoPi * gamma[source] / denominator;
            velocityX -= coefficient * dy;
            velocityY += coefficient * dx;
        } else if (geometry == Geometry::periodic_x) {
            if (source == target)
                continue;
            const double scaledX = waveNumber * remainder(x[target] - x[source], first);
            const double scaledY = waveNumber * (y[target] - y[source]);
            const double magnitudeY = fabs(scaledY);
            double sinhRatio = 0.0, sineRatio = 0.0;
            if (magnitudeY <= 40.0) {
                const double sinhHalfY = sinh(0.5 * scaledY);
                const double sinHalfX = sin(0.5 * scaledX);
                const double denominator = 2.0 * (sinhHalfY * sinhHalfY + sinHalfX * sinHalfX);
                if (denominator == 0.0) {
                    atomicExch(failure, 1);
                    continue;
                }
                sinhRatio = sinh(scaledY) / denominator;
                sineRatio = sin(scaledX) / denominator;
            } else {
                const double q = exp(-magnitudeY);
                const double denominator = 1.0 + q * q - 2.0 * q * cos(scaledX);
                sinhRatio = copysign((1.0 - q * q) / denominator, scaledY);
                sineRatio = 2.0 * q * sin(scaledX) / denominator;
            }
            velocityX -= periodicScale * gamma[source] * sinhRatio;
            velocityY += periodicScale * gamma[source] * sineRatio;
        } else if (geometry == Geometry::periodic) {
            // The symmetric nonzero self images have zero sine numerators.
            if (source == target)
                continue;
            const double dx = waveNumber * remainder(x[target] - x[source], first);
            const double dy = waveNumber * remainder(y[target] - y[source], second);
            const double sineX = sin(dx), sineY = sin(dy);
            const double sinHalfX = sin(0.5 * dx), sinHalfY = sin(0.5 * dy);
            for (int image = -imageLayers; image <= imageLayers; ++image) {
                const double shiftedX = dx - twoPi * image;
                const double shiftedY = dy - twoPi * image;
                const double sinhHalfX = fabs(shiftedX) > 40.0 ? CUDART_INF : sinh(0.5 * shiftedX);
                const double sinhHalfY = fabs(shiftedY) > 40.0 ? CUDART_INF : sinh(0.5 * shiftedY);
                const double denominatorU = 2.0 * (sinhHalfX * sinhHalfX + sinHalfY * sinHalfY);
                const double denominatorV = 2.0 * (sinhHalfY * sinhHalfY + sinHalfX * sinHalfX);
                if (denominatorU == 0.0 || denominatorV == 0.0) {
                    atomicExch(failure, 1);
                    continue;
                }
                velocityX -= periodicScale * gamma[source] * sineY / denominatorU;
                velocityY += periodicScale * gamma[source] * sineX / denominatorV;
            }
        } else {
            if (source != target) {
                const double dx = x[target] - x[source];
                const double dy = y[target] - y[source];
                const double denominator = dx * dx + dy * dy;
                if (denominator == 0.0) {
                    atomicExch(failure, 1);
                } else {
                    const double coefficient = inverseTwoPi * gamma[source] / denominator;
                    velocityX -= coefficient * dy;
                    velocityY += coefficient * dx;
                }
            }
            const double sx = x[source] * inverseRadius, sy = y[source] * inverseRadius;
            const double a = 1.0 - (targetX * sx + targetY * sy);
            const double b = targetY * sx - targetX * sy;
            const double denominator = a * a + b * b;
            const double imageX = -a * sx - b * sy;
            const double imageY = -a * sy + b * sx;
            const double coefficient = -inverseTwoPi * gamma[source] * inverseRadius / denominator;
            velocityX -= coefficient * imageY;
            velocityY += coefficient * imageX;
        }
    }
    if (!isfinite(velocityX) || !isfinite(velocityY))
        atomicExch(failure, 1);
    else {
        u[target] = velocityX;
        v[target] = velocityY;
    }
}

__global__ void validateStateKernel(const double *x, const double *y, std::size_t count,
                                    Geometry geometry, double diskRadiusSquared, int *failure) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const double px = x[index], py = y[index];
    if (!isfinite(px) || !isfinite(py) ||
        (geometry == Geometry::disk && px * px + py * py >= diskRadiusSquared))
        atomicExch(failure, 1);
}

__global__ void makeStageKernel(double *x, double *y, const double *initialX,
                                const double *initialY, const double *const *stageX,
                                const double *const *stageY, std::size_t count, double dt,
                                double c0, double c1, double c2, double c3, double c4, double c5,
                                double c6) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const double coefficients[7] = {c0, c1, c2, c3, c4, c5, c6};
    double dx = 0.0, dy = 0.0;
    for (int stage = 0; stage < 7; ++stage) {
        dx += coefficients[stage] * stageX[stage][index];
        dy += coefficients[stage] * stageY[stage][index];
    }
    x[index] = initialX[index] + dt * dx;
    y[index] = initialY[index] + dt * dy;
}

__global__ void rk4CombineKernel(double *x, double *y, const double *initialX,
                                 const double *initialY, const double *const *stageX,
                                 const double *const *stageY, std::size_t count, double dt) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    x[index] = initialX[index] + dt *
                                     (stageX[0][index] + 2.0 * stageX[1][index] +
                                      2.0 * stageX[2][index] + stageX[3][index]) /
                                     6.0;
    y[index] = initialY[index] + dt *
                                     (stageY[0][index] + 2.0 * stageY[1][index] +
                                      2.0 * stageY[2][index] + stageY[3][index]) /
                                     6.0;
}

__global__ void dopriErrorKernel(const double *initialX, const double *initialY,
                                 const double *candidateX, const double *candidateY,
                                 const double *const *stageX, const double *const *stageY,
                                 std::size_t count, double dt, double absoluteTolerance,
                                 double relativeTolerance, double *blockErrors) {
    __shared__ double maximum[256];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (index < count) {
        constexpr double weights[7] = {35.0 / 384.0 - 5179.0 / 57600.0,
                                       0.0,
                                       500.0 / 1113.0 - 7571.0 / 16695.0,
                                       125.0 / 192.0 - 393.0 / 640.0,
                                       -2187.0 / 6784.0 + 92097.0 / 339200.0,
                                       11.0 / 84.0 - 187.0 / 2100.0,
                                       -1.0 / 40.0};
        double errorX = 0.0, errorY = 0.0;
        for (int stage = 0; stage < 7; ++stage) {
            errorX += dt * weights[stage] * stageX[stage][index];
            errorY += dt * weights[stage] * stageY[stage][index];
        }
        const double scaleX = absoluteTolerance + relativeTolerance * fmax(fabs(initialX[index]),
                                                                           fabs(candidateX[index]));
        const double scaleY = absoluteTolerance + relativeTolerance * fmax(fabs(initialY[index]),
                                                                           fabs(candidateY[index]));
        if (!isfinite(errorX) || !isfinite(errorY) || !isfinite(scaleX) || !isfinite(scaleY) ||
            !(scaleX > 0.0) || !(scaleY > 0.0))
            local = CUDART_INF;
        else
            local = fmax(fabs(errorX) / scaleX, fabs(errorY) / scaleY);
    }
    maximum[threadIdx.x] = local;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            maximum[threadIdx.x] = fmax(maximum[threadIdx.x], maximum[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        blockErrors[blockIdx.x] = maximum[0];
}

class CudaKernel final : public VelocityKernel {
  public:
    explicit CudaKernel(const SimParams &params)
        : params_(params), cpu_(makeReferenceKernel(params)) {
        if (params.boundaryCondition == "infinite")
            geometry_ = Geometry::infinite;
        else if (params.boundaryCondition == "periodic_x")
            geometry_ = Geometry::periodic_x;
        else if (params.boundaryCondition == "periodic")
            geometry_ = Geometry::periodic;
        else
            geometry_ = Geometry::disk;
    }
    ~CudaKernel() override { release(); }

    void evaluateRange(const std::vector<double> &x, const std::vector<double> &y,
                       const std::vector<double> &gamma, VelocityField &velocity, std::size_t begin,
                       std::size_t end) const override {
        validateVortexArrays(x, y, gamma);
        const std::size_t count = x.size();
        if (begin > end || end > count)
            throw std::invalid_argument("invalid CUDA vortex arrays or target range");
        validateGeometry(x, y, gamma);

        velocity.resize(count);
        if (begin == end)
            return;
        const std::size_t bytes = count * sizeof(double);
        ensureCapacity(count);
        stateCount_ = count;
        deviceStateValid_ = true;
        fsalValid_ = false;
        cudaCheck(cudaMemcpy(deviceX_, x.data(), bytes, cudaMemcpyHostToDevice), "copy x to GPU");
        cudaCheck(cudaMemcpy(deviceY_, y.data(), bytes, cudaMemcpyHostToDevice), "copy y to GPU");
        cudaCheck(cudaMemcpy(deviceGamma_, gamma.data(), bytes, cudaMemcpyHostToDevice),
                  "copy circulation to GPU");
        evaluateDevice(deviceX_, deviceY_, deviceU_, deviceV_, begin, end);
        cudaCheck(cudaMemcpy(velocity.x.data() + begin, deviceU_ + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy u from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data() + begin, deviceV_ + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy v from GPU");
    }

    double hamiltonian(const VortexSystem &state) const override {
        return cpu_->hamiltonian(state);
    }
    bool supportsDeviceStepping() const noexcept override { return true; }
    void uploadDeviceState(const VortexSystem &state) const override {
        state.validate();
        validateGeometry(state.x, state.y, state.circulation);
        if (state.size() == 0) {
            stateCount_ = 0;
            deviceStateValid_ = true;
            fsalValid_ = false;
            return;
        }
        ensureCapacity(state.size());
        stateCount_ = state.size();
        fsalValid_ = false;
        deviceStateValid_ = true;
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(deviceX_, state.x.data(), bytes, cudaMemcpyHostToDevice),
                  "upload state x to GPU");
        cudaCheck(cudaMemcpy(deviceY_, state.y.data(), bytes, cudaMemcpyHostToDevice),
                  "upload state y to GPU");
        cudaCheck(cudaMemcpy(deviceGamma_, state.circulation.data(), bytes, cudaMemcpyHostToDevice),
                  "upload circulation to GPU");
    }
    void downloadDeviceState(VortexSystem &state) const override {
        requireDeviceState();
        if (state.size() != stateCount_)
            throw std::runtime_error("host and CUDA vortex populations differ");
        if (stateCount_ == 0)
            return;
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(state.x.data(), deviceX_, bytes, cudaMemcpyDeviceToHost),
                  "download state x from GPU");
        cudaCheck(cudaMemcpy(state.y.data(), deviceY_, bytes, cudaMemcpyDeviceToHost),
                  "download state y from GPU");
        state.validate();
    }
    void evaluateDeviceState(VelocityField &velocity) const override {
        requireDeviceState();
        velocity.resize(stateCount_);
        if (stateCount_ == 0)
            return;
        evaluateDevice(deviceX_, deviceY_, deviceU_, deviceV_, 0, stateCount_);
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(velocity.x.data(), deviceU_, bytes, cudaMemcpyDeviceToHost),
                  "download velocity x from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data(), deviceV_, bytes, cudaMemcpyDeviceToHost),
                  "download velocity y from GPU");
        for (std::size_t i = 0; i < stateCount_; ++i)
            if (!std::isfinite(velocity.x[i]) || !std::isfinite(velocity.y[i]))
                throw std::runtime_error(
                    "non-finite CUDA velocity; check scales and close encounters");
    }
    void deviceRk4Step(double dt) const override {
        if (!std::isfinite(dt) || !(dt > 0.0))
            throw std::invalid_argument("timestep must be finite and positive");
        requireDeviceState();
        if (stateCount_ == 0)
            return;
        copyStateToInitial();
        fsalValid_ = false;
        evaluateDevice(deviceX_, deviceY_, deviceStageX_[0], deviceStageY_[0], 0, stateCount_);
        makeStage(0.5 * dt, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_, deviceTemporaryY_, deviceStageX_[1], deviceStageY_[1], 0,
                       stateCount_);
        makeStage(0.5 * dt, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_, deviceTemporaryY_, deviceStageX_[2], deviceStageY_[2], 0,
                       stateCount_);
        makeStage(dt, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_, deviceTemporaryY_, deviceStageX_[3], deviceStageY_[3], 0,
                       stateCount_);
        const int blocks = blockCount(stateCount_);
        rk4CombineKernel<<<blocks, threadsPerBlock>>>(deviceX_, deviceY_, deviceInitialX_,
                                                      deviceInitialY_, deviceStageXPtrs_,
                                                      deviceStageYPtrs_, stateCount_, dt);
        cudaCheck(cudaGetLastError(), "launch CUDA RK4 final stage");
        validateDeviceState(deviceX_, deviceY_);
    }
    StepResult deviceDopri5Step(double dt, double absoluteTolerance, double relativeTolerance,
                                double minimumTimeStep, double maximumTimeStep) const override {
        if (!std::isfinite(dt) || !(dt > 0.0))
            throw std::invalid_argument("timestep must be finite and positive");
        requireDeviceState();
        if (stateCount_ == 0)
            return {dt, std::min(maximumTimeStep, 5.0 * dt), 0.0, 0};
        copyStateToInitial();
        if (!fsalValid_)
            evaluateDevice(deviceX_, deviceY_, deviceStageX_[0], deviceStageY_[0], 0, stateCount_);
        unsigned rejected = 0;
        for (;;) {
            makeDopriStages(dt);
            const double error = dopriError(dt, absoluteTolerance, relativeTolerance);
            const double factor =
                error == 0.0 ? 5.0 : std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0);
            const double suggested = std::clamp(dt * factor, minimumTimeStep, maximumTimeStep);
            if (error <= 1.0) {
                const std::size_t bytes = stateCount_ * sizeof(double);
                cudaCheck(cudaMemcpy(deviceX_, deviceTemporaryX_, bytes, cudaMemcpyDeviceToDevice),
                          "accept CUDA DOPRI5 x state");
                cudaCheck(cudaMemcpy(deviceY_, deviceTemporaryY_, bytes, cudaMemcpyDeviceToDevice),
                          "accept CUDA DOPRI5 y state");
                std::swap(deviceStageX_[0], deviceStageX_[6]);
                std::swap(deviceStageY_[0], deviceStageY_[6]);
                refreshStagePointers();
                fsalValid_ = true;
                return {dt, suggested, error, rejected};
            }
            if (dt <= minimumTimeStep || ++rejected > 32)
                throw std::runtime_error("adaptive CUDA integrator could not satisfy tolerance");
            dt = std::max(minimumTimeStep, std::min(suggested, dt * 0.9));
        }
    }
    void invalidateDeviceDerivative() const noexcept override { fsalValid_ = false; }

  private:
    static constexpr int threadsPerBlock = 256;
    int blockCount(std::size_t count) const {
        const std::size_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CUDA grid exceeds the supported block count");
        return static_cast<int>(blocks);
    }
    void validateGeometry(const std::vector<double> &x, const std::vector<double> &y,
                          const std::vector<double> &circulation) const {
        if (geometry_ == Geometry::periodic)
            validatePeriodicCirculation(circulation);
        if (geometry_ == Geometry::disk)
            validateDiskPositions(x, y, params_.diskRadius * params_.diskRadius);
    }
    void requireDeviceState() const {
        if (!deviceStateValid_)
            throw std::logic_error("CUDA device state has not been uploaded");
    }
    void refreshStagePointers() const {
        cudaCheck(cudaMemcpy(deviceStageXPtrs_, deviceStageX_.data(),
                             deviceStageX_.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA x-stage pointers");
        cudaCheck(cudaMemcpy(deviceStageYPtrs_, deviceStageY_.data(),
                             deviceStageY_.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA y-stage pointers");
    }
    void checkFailure(const char *operation) const {
        int failure = 0;
        cudaCheck(cudaMemcpy(&failure, deviceFailure_, sizeof(int), cudaMemcpyDeviceToHost),
                  "read CUDA state-validation flag");
        if (failure)
            throw std::runtime_error(std::string(operation) +
                                     ": non-finite, coincident, or out-of-domain vortex state");
    }
    void validateDeviceState(const double *x, const double *y) const {
        cudaCheck(cudaMemset(deviceFailure_, 0, sizeof(int)), "clear CUDA state-validation flag");
        validateStateKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            x, y, stateCount_, geometry_, params_.diskRadius * params_.diskRadius, deviceFailure_);
        cudaCheck(cudaGetLastError(), "launch CUDA state-validation kernel");
        checkFailure("CUDA state validation failed");
    }
    void evaluateDevice(const double *x, const double *y, double *u, double *v, std::size_t begin,
                        std::size_t end) const {
        if (begin == end)
            return;
        cudaCheck(cudaMemset(deviceFailure_, 0, sizeof(int)), "clear CUDA velocity error flag");
        const bool periodicX = geometry_ == Geometry::periodic_x || geometry_ == Geometry::periodic;
        const double first = geometry_ == Geometry::infinite
                                 ? params_.coreRadius * params_.coreRadius
                                 : (periodicX ? params_.boxLengthX : params_.diskRadius);
        velocityKernel<<<blockCount(end - begin), threadsPerBlock>>>(
            x, y, deviceGamma_, u, v, stateCount_, begin, end, geometry_, first, params_.boxLengthY,
            params_.periodicImageLayers, deviceFailure_);
        cudaCheck(cudaGetLastError(), "launch CUDA velocity kernel");
        checkFailure("CUDA velocity evaluation failed");
    }
    void copyStateToInitial() const {
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(deviceInitialX_, deviceX_, bytes, cudaMemcpyDeviceToDevice),
                  "copy CUDA initial x state");
        cudaCheck(cudaMemcpy(deviceInitialY_, deviceY_, bytes, cudaMemcpyDeviceToDevice),
                  "copy CUDA initial y state");
    }
    void makeStage(double dt, const std::array<double, 7> &coefficients) const {
        makeStageKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            deviceTemporaryX_, deviceTemporaryY_, deviceInitialX_, deviceInitialY_,
            deviceStageXPtrs_, deviceStageYPtrs_, stateCount_, dt, coefficients[0], coefficients[1],
            coefficients[2], coefficients[3], coefficients[4], coefficients[5], coefficients[6]);
        cudaCheck(cudaGetLastError(), "launch CUDA Runge--Kutta stage");
    }
    void makeDopriStages(double dt) const {
        for (std::size_t stage = 1; stage < 7; ++stage) {
            makeStage(dt, integrator_detail::dopriCoefficients[stage]);
            evaluateDevice(deviceTemporaryX_, deviceTemporaryY_, deviceStageX_[stage],
                           deviceStageY_[stage], 0, stateCount_);
        }
    }
    double dopriError(double dt, double absoluteTolerance, double relativeTolerance) const {
        const int blocks = blockCount(stateCount_);
        dopriErrorKernel<<<blocks, threadsPerBlock>>>(
            deviceInitialX_, deviceInitialY_, deviceTemporaryX_, deviceTemporaryY_,
            deviceStageXPtrs_, deviceStageYPtrs_, stateCount_, dt, absoluteTolerance,
            relativeTolerance, deviceBlockErrors_);
        cudaCheck(cudaGetLastError(), "launch CUDA DOPRI5 error kernel");
        hostBlockErrors_.resize(static_cast<std::size_t>(blocks));
        cudaCheck(cudaMemcpy(hostBlockErrors_.data(), deviceBlockErrors_,
                             hostBlockErrors_.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "download CUDA DOPRI5 error blocks");
        return *std::max_element(hostBlockErrors_.begin(), hostBlockErrors_.end());
    }
    void ensureCapacity(std::size_t count) const {
        if (count <= capacity_)
            return;
        release();
        const std::size_t bytes = count * sizeof(double);
        try {
            cudaCheck(cudaMalloc(&deviceX_, bytes), "cudaMalloc(x)");
            cudaCheck(cudaMalloc(&deviceY_, bytes), "cudaMalloc(y)");
            cudaCheck(cudaMalloc(&deviceGamma_, bytes), "cudaMalloc(circulation)");
            cudaCheck(cudaMalloc(&deviceU_, bytes), "cudaMalloc(u)");
            cudaCheck(cudaMalloc(&deviceV_, bytes), "cudaMalloc(v)");
            cudaCheck(cudaMalloc(&deviceInitialX_, bytes), "cudaMalloc(initial x)");
            cudaCheck(cudaMalloc(&deviceInitialY_, bytes), "cudaMalloc(initial y)");
            cudaCheck(cudaMalloc(&deviceTemporaryX_, bytes), "cudaMalloc(temporary x)");
            cudaCheck(cudaMalloc(&deviceTemporaryY_, bytes), "cudaMalloc(temporary y)");
            for (std::size_t stage = 0; stage < deviceStageX_.size(); ++stage) {
                cudaCheck(cudaMalloc(&deviceStageX_[stage], bytes), "cudaMalloc(x stage)");
                cudaCheck(cudaMalloc(&deviceStageY_[stage], bytes), "cudaMalloc(y stage)");
            }
            cudaCheck(cudaMalloc(&deviceStageXPtrs_, deviceStageX_.size() * sizeof(double *)),
                      "cudaMalloc(x-stage pointers)");
            cudaCheck(cudaMalloc(&deviceStageYPtrs_, deviceStageY_.size() * sizeof(double *)),
                      "cudaMalloc(y-stage pointers)");
            cudaCheck(
                cudaMalloc(&deviceBlockErrors_,
                           ((count + threadsPerBlock - 1) / threadsPerBlock) * sizeof(double)),
                "cudaMalloc(DOPRI5 errors)");
            cudaCheck(cudaMalloc(&deviceFailure_, sizeof(int)), "cudaMalloc(error flag)");
            refreshStagePointers();
            capacity_ = count;
        } catch (...) {
            release();
            throw;
        }
    }
    void release() const noexcept {
        cudaFree(deviceX_);
        cudaFree(deviceY_);
        cudaFree(deviceGamma_);
        cudaFree(deviceU_);
        cudaFree(deviceV_);
        cudaFree(deviceInitialX_);
        cudaFree(deviceInitialY_);
        cudaFree(deviceTemporaryX_);
        cudaFree(deviceTemporaryY_);
        for (double *&stage : deviceStageX_)
            cudaFree(stage);
        for (double *&stage : deviceStageY_)
            cudaFree(stage);
        cudaFree(deviceStageXPtrs_);
        cudaFree(deviceStageYPtrs_);
        cudaFree(deviceBlockErrors_);
        cudaFree(deviceFailure_);
        deviceX_ = deviceY_ = deviceGamma_ = deviceU_ = deviceV_ = nullptr;
        deviceInitialX_ = deviceInitialY_ = deviceTemporaryX_ = deviceTemporaryY_ = nullptr;
        deviceStageX_.fill(nullptr);
        deviceStageY_.fill(nullptr);
        deviceStageXPtrs_ = deviceStageYPtrs_ = nullptr;
        deviceBlockErrors_ = nullptr;
        deviceFailure_ = nullptr;
        capacity_ = 0;
        stateCount_ = 0;
        deviceStateValid_ = false;
        fsalValid_ = false;
    }
    SimParams params_;
    Geometry geometry_ = Geometry::infinite;
    std::unique_ptr<VelocityKernel> cpu_;
    mutable double *deviceX_ = nullptr, *deviceY_ = nullptr, *deviceGamma_ = nullptr;
    mutable double *deviceU_ = nullptr, *deviceV_ = nullptr;
    mutable double *deviceInitialX_ = nullptr, *deviceInitialY_ = nullptr;
    mutable double *deviceTemporaryX_ = nullptr, *deviceTemporaryY_ = nullptr;
    mutable std::array<double *, 7> deviceStageX_{};
    mutable std::array<double *, 7> deviceStageY_{};
    mutable double **deviceStageXPtrs_ = nullptr, **deviceStageYPtrs_ = nullptr;
    mutable double *deviceBlockErrors_ = nullptr;
    mutable int *deviceFailure_ = nullptr;
    mutable std::vector<double> hostBlockErrors_;
    mutable std::size_t capacity_ = 0;
    mutable std::size_t stateCount_ = 0;
    mutable bool deviceStateValid_ = false;
    mutable bool fsalValid_ = false;
};
} // namespace

void backendInitialize(int &, char **&) { cudaCheck(cudaFree(nullptr), "initialize CUDA"); }
void backendFinalize() {}
void backendAbort(int) {}
bool backendIsRoot() { return true; }
const char *backendName() { return "CUDA"; }
std::string backendRuntimeDetails() {
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "query CUDA device");
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, device), "query CUDA device properties");
    return "cuda_device " + std::to_string(device) + "\ncuda_device_name \"" +
           std::string(properties.name) + "\"\ncuda_compute_capability " +
           std::to_string(properties.major) + "." + std::to_string(properties.minor);
}
std::unique_ptr<VelocityKernel> makeBackendKernel(const SimParams &params) {
    return std::make_unique<CudaKernel>(params);
}
