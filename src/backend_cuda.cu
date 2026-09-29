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
#include <utility>
#include <vector>

namespace {

void cudaCheck(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T> class CudaBuffer {
  public:
    CudaBuffer() = default;
    ~CudaBuffer() { reset(); }
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    CudaBuffer &operator=(CudaBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count, const char *operation) {
        reset();
        cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), operation);
        size_ = count;
    }
    void reset() noexcept {
        cudaFree(data_);
        data_ = nullptr;
        size_ = 0;
    }
    [[nodiscard]] T *get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

  private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};

struct DeviceGeometry {
    BoundaryKind boundary;
    double coreRadiusSquared;
    double boxLengthX;
    double boxLengthY;
    double diskRadius;
    int periodicImageLayers;
};

__global__ void velocityKernel(const double *x, const double *y, const double *gamma, double *u,
                               double *v, std::size_t count, std::size_t begin, std::size_t end,
                               DeviceGeometry geometry, int *failure) {
    const std::size_t target =
        begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (target >= end)
        return;
    if (!isfinite(x[target]) || !isfinite(y[target]) ||
        (geometry.boundary == BoundaryKind::disk &&
         x[target] * x[target] + y[target] * y[target] >=
             geometry.diskRadius * geometry.diskRadius)) {
        atomicExch(failure, 1);
        return;
    }
    constexpr double inverseTwoPi = 0.15915494309189533576888376337251;
    constexpr double twoPi = 6.283185307179586476925286766559;
    double velocityX = 0.0, velocityY = 0.0;
    double inverseRadius = 0.0, targetX = 0.0, targetY = 0.0;
    double waveNumber = 0.0, periodicScale = 0.0;
    if (geometry.boundary == BoundaryKind::periodic_x ||
        geometry.boundary == BoundaryKind::periodic) {
        waveNumber = twoPi / geometry.boxLengthX;
        periodicScale = 0.5 / geometry.boxLengthX;
    }
    if (geometry.boundary == BoundaryKind::disk) {
        inverseRadius = 1.0 / geometry.diskRadius;
        targetX = x[target] * inverseRadius;
        targetY = y[target] * inverseRadius;
    }

    for (std::size_t source = 0; source < count; ++source) {
        if (geometry.boundary == BoundaryKind::infinite) {
            if (source == target)
                continue;
            const double dx = x[target] - x[source];
            const double dy = y[target] - y[source];
            const double denominator = dx * dx + dy * dy + geometry.coreRadiusSquared;
            if (denominator == 0.0) {
                atomicExch(failure, 1);
                continue;
            }
            const double coefficient = inverseTwoPi * gamma[source] / denominator;
            velocityX -= coefficient * dy;
            velocityY += coefficient * dx;
        } else if (geometry.boundary == BoundaryKind::periodic_x) {
            if (source == target)
                continue;
            const double scaledX =
                waveNumber * remainder(x[target] - x[source], geometry.boxLengthX);
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
        } else if (geometry.boundary == BoundaryKind::periodic) {
            // The symmetric nonzero self images have zero sine numerators.
            if (source == target)
                continue;
            const double dx = waveNumber * remainder(x[target] - x[source], geometry.boxLengthX);
            const double dy = waveNumber * remainder(y[target] - y[source], geometry.boxLengthY);
            const double sineX = sin(dx), sineY = sin(dy);
            const double sinHalfX = sin(0.5 * dx), sinHalfY = sin(0.5 * dy);
            for (int image = -geometry.periodicImageLayers; image <= geometry.periodicImageLayers;
                 ++image) {
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
                                    BoundaryKind boundary, double diskRadiusSquared, int *failure) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const double px = x[index], py = y[index];
    if (!isfinite(px) || !isfinite(py) ||
        (boundary == BoundaryKind::disk && px * px + py * py >= diskRadiusSquared))
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

class CudaKernel final : public VelocityKernel, public DeviceStepper {
  public:
    explicit CudaKernel(const SimParams &params)
        : params_(params), geometry_{params.boundary,   params.coreRadius * params.coreRadius,
                                     params.boxLengthX, params.boxLengthY,
                                     params.diskRadius, params.periodicImageLayers},
          cpu_(makeReferenceKernel(params)) {}

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
        cudaCheck(cudaMemcpy(deviceX_.get(), x.data(), bytes, cudaMemcpyHostToDevice),
                  "copy x to GPU");
        cudaCheck(cudaMemcpy(deviceY_.get(), y.data(), bytes, cudaMemcpyHostToDevice),
                  "copy y to GPU");
        cudaCheck(cudaMemcpy(deviceGamma_.get(), gamma.data(), bytes, cudaMemcpyHostToDevice),
                  "copy circulation to GPU");
        evaluateDevice(deviceX_.get(), deviceY_.get(), deviceU_.get(), deviceV_.get(), begin, end);
        cudaCheck(cudaMemcpy(velocity.x.data() + begin, deviceU_.get() + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy u from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data() + begin, deviceV_.get() + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy v from GPU");
    }

    double hamiltonian(const VortexSystem &state) const override {
        return cpu_->hamiltonian(state);
    }
    void uploadState(const VortexSystem &state) const override {
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
        cudaCheck(cudaMemcpy(deviceX_.get(), state.x.data(), bytes, cudaMemcpyHostToDevice),
                  "upload state x to GPU");
        cudaCheck(cudaMemcpy(deviceY_.get(), state.y.data(), bytes, cudaMemcpyHostToDevice),
                  "upload state y to GPU");
        cudaCheck(
            cudaMemcpy(deviceGamma_.get(), state.circulation.data(), bytes, cudaMemcpyHostToDevice),
            "upload circulation to GPU");
    }
    void downloadState(VortexSystem &state) const override {
        requireDeviceState();
        if (state.size() != stateCount_)
            throw std::runtime_error("host and CUDA vortex populations differ");
        if (stateCount_ == 0)
            return;
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(state.x.data(), deviceX_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download state x from GPU");
        cudaCheck(cudaMemcpy(state.y.data(), deviceY_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download state y from GPU");
        state.validate();
    }
    void evaluateState(VelocityField &velocity) const override {
        requireDeviceState();
        velocity.resize(stateCount_);
        if (stateCount_ == 0)
            return;
        evaluateDevice(deviceX_.get(), deviceY_.get(), deviceU_.get(), deviceV_.get(), 0,
                       stateCount_);
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(velocity.x.data(), deviceU_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download velocity x from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data(), deviceV_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download velocity y from GPU");
        for (std::size_t i = 0; i < stateCount_; ++i)
            if (!std::isfinite(velocity.x[i]) || !std::isfinite(velocity.y[i]))
                throw std::runtime_error(
                    "non-finite CUDA velocity; check scales and close encounters");
    }
    void rk4Step(double dt) const override {
        if (!std::isfinite(dt) || !(dt > 0.0))
            throw std::invalid_argument("timestep must be finite and positive");
        requireDeviceState();
        if (stateCount_ == 0)
            return;
        copyStateToInitial();
        fsalValid_ = false;
        evaluateDevice(deviceX_.get(), deviceY_.get(), deviceStageX_[0].get(),
                       deviceStageY_[0].get(), 0, stateCount_);
        makeStage(0.5 * dt, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_.get(), deviceTemporaryY_.get(), deviceStageX_[1].get(),
                       deviceStageY_[1].get(), 0, stateCount_);
        makeStage(0.5 * dt, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_.get(), deviceTemporaryY_.get(), deviceStageX_[2].get(),
                       deviceStageY_[2].get(), 0, stateCount_);
        makeStage(dt, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        evaluateDevice(deviceTemporaryX_.get(), deviceTemporaryY_.get(), deviceStageX_[3].get(),
                       deviceStageY_[3].get(), 0, stateCount_);
        const int blocks = blockCount(stateCount_);
        rk4CombineKernel<<<blocks, threadsPerBlock>>>(
            deviceX_.get(), deviceY_.get(), deviceInitialX_.get(), deviceInitialY_.get(),
            deviceStageXPtrs_.get(), deviceStageYPtrs_.get(), stateCount_, dt);
        cudaCheck(cudaGetLastError(), "launch CUDA RK4 final stage");
        validateDeviceState(deviceX_.get(), deviceY_.get());
    }
    StepResult dopri5Step(double dt, double absoluteTolerance, double relativeTolerance,
                          double minimumTimeStep, double maximumTimeStep) const override {
        if (!std::isfinite(dt) || !(dt > 0.0))
            throw std::invalid_argument("timestep must be finite and positive");
        requireDeviceState();
        if (stateCount_ == 0)
            return {dt, std::min(maximumTimeStep, 5.0 * dt), 0.0, 0};
        copyStateToInitial();
        if (!fsalValid_)
            evaluateDevice(deviceX_.get(), deviceY_.get(), deviceStageX_[0].get(),
                           deviceStageY_[0].get(), 0, stateCount_);
        unsigned rejected = 0;
        for (;;) {
            makeDopriStages(dt);
            const double error = dopriError(dt, absoluteTolerance, relativeTolerance);
            const double factor =
                error == 0.0 ? 5.0 : std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0);
            const double suggested = std::clamp(dt * factor, minimumTimeStep, maximumTimeStep);
            if (error <= 1.0) {
                const std::size_t bytes = stateCount_ * sizeof(double);
                cudaCheck(cudaMemcpy(deviceX_.get(), deviceTemporaryX_.get(), bytes,
                                     cudaMemcpyDeviceToDevice),
                          "accept CUDA DOPRI5 x state");
                cudaCheck(cudaMemcpy(deviceY_.get(), deviceTemporaryY_.get(), bytes,
                                     cudaMemcpyDeviceToDevice),
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
    void invalidateDerivative() const noexcept override { fsalValid_ = false; }

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
        if (isPeriodicY(geometry_.boundary))
            validatePeriodicCirculation(circulation);
        if (isDisk(geometry_.boundary))
            validateDiskPositions(x, y, params_.diskRadius * params_.diskRadius);
    }
    void requireDeviceState() const {
        if (!deviceStateValid_)
            throw std::logic_error("CUDA device state has not been uploaded");
    }
    void refreshStagePointers() const {
        std::array<double *, 7> stageXPointers{}, stageYPointers{};
        for (std::size_t stage = 0; stage < deviceStageX_.size(); ++stage) {
            stageXPointers[stage] = deviceStageX_[stage].get();
            stageYPointers[stage] = deviceStageY_[stage].get();
        }
        cudaCheck(cudaMemcpy(deviceStageXPtrs_.get(), stageXPointers.data(),
                             stageXPointers.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA x-stage pointers");
        cudaCheck(cudaMemcpy(deviceStageYPtrs_.get(), stageYPointers.data(),
                             stageYPointers.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA y-stage pointers");
    }
    void checkFailure(const char *operation) const {
        int failure = 0;
        cudaCheck(cudaMemcpy(&failure, deviceFailure_.get(), sizeof(int), cudaMemcpyDeviceToHost),
                  "read CUDA state-validation flag");
        if (failure)
            throw std::runtime_error(std::string(operation) +
                                     ": non-finite, coincident, or out-of-domain vortex state");
    }
    void validateDeviceState(const double *x, const double *y) const {
        cudaCheck(cudaMemset(deviceFailure_.get(), 0, sizeof(int)),
                  "clear CUDA state-validation flag");
        validateStateKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            x, y, stateCount_, geometry_.boundary, geometry_.diskRadius * geometry_.diskRadius,
            deviceFailure_.get());
        cudaCheck(cudaGetLastError(), "launch CUDA state-validation kernel");
        checkFailure("CUDA state validation failed");
    }
    void evaluateDevice(const double *x, const double *y, double *u, double *v, std::size_t begin,
                        std::size_t end) const {
        if (begin == end)
            return;
        cudaCheck(cudaMemset(deviceFailure_.get(), 0, sizeof(int)),
                  "clear CUDA velocity error flag");
        velocityKernel<<<blockCount(end - begin), threadsPerBlock>>>(
            x, y, deviceGamma_.get(), u, v, stateCount_, begin, end, geometry_,
            deviceFailure_.get());
        cudaCheck(cudaGetLastError(), "launch CUDA velocity kernel");
        checkFailure("CUDA velocity evaluation failed");
    }
    void copyStateToInitial() const {
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(
            cudaMemcpy(deviceInitialX_.get(), deviceX_.get(), bytes, cudaMemcpyDeviceToDevice),
            "copy CUDA initial x state");
        cudaCheck(
            cudaMemcpy(deviceInitialY_.get(), deviceY_.get(), bytes, cudaMemcpyDeviceToDevice),
            "copy CUDA initial y state");
    }
    void makeStage(double dt, const std::array<double, 7> &coefficients) const {
        makeStageKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            deviceTemporaryX_.get(), deviceTemporaryY_.get(), deviceInitialX_.get(),
            deviceInitialY_.get(), deviceStageXPtrs_.get(), deviceStageYPtrs_.get(), stateCount_,
            dt, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
            coefficients[5], coefficients[6]);
        cudaCheck(cudaGetLastError(), "launch CUDA Runge--Kutta stage");
    }
    void makeDopriStages(double dt) const {
        for (std::size_t stage = 1; stage < 7; ++stage) {
            makeStage(dt, integrator_detail::dopriCoefficients[stage]);
            evaluateDevice(deviceTemporaryX_.get(), deviceTemporaryY_.get(),
                           deviceStageX_[stage].get(), deviceStageY_[stage].get(), 0, stateCount_);
        }
    }
    double dopriError(double dt, double absoluteTolerance, double relativeTolerance) const {
        const int blocks = blockCount(stateCount_);
        dopriErrorKernel<<<blocks, threadsPerBlock>>>(
            deviceInitialX_.get(), deviceInitialY_.get(), deviceTemporaryX_.get(),
            deviceTemporaryY_.get(), deviceStageXPtrs_.get(), deviceStageYPtrs_.get(), stateCount_,
            dt, absoluteTolerance, relativeTolerance, deviceBlockErrors_.get());
        cudaCheck(cudaGetLastError(), "launch CUDA DOPRI5 error kernel");
        hostBlockErrors_.resize(static_cast<std::size_t>(blocks));
        cudaCheck(cudaMemcpy(hostBlockErrors_.data(), deviceBlockErrors_.get(),
                             hostBlockErrors_.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "download CUDA DOPRI5 error blocks");
        return *std::max_element(hostBlockErrors_.begin(), hostBlockErrors_.end());
    }
    void ensureCapacity(std::size_t count) const {
        if (count <= capacity_)
            return;
        release();
        try {
            deviceX_.allocate(count, "cudaMalloc(x)");
            deviceY_.allocate(count, "cudaMalloc(y)");
            deviceGamma_.allocate(count, "cudaMalloc(circulation)");
            deviceU_.allocate(count, "cudaMalloc(u)");
            deviceV_.allocate(count, "cudaMalloc(v)");
            deviceInitialX_.allocate(count, "cudaMalloc(initial x)");
            deviceInitialY_.allocate(count, "cudaMalloc(initial y)");
            deviceTemporaryX_.allocate(count, "cudaMalloc(temporary x)");
            deviceTemporaryY_.allocate(count, "cudaMalloc(temporary y)");
            for (std::size_t stage = 0; stage < deviceStageX_.size(); ++stage) {
                deviceStageX_[stage].allocate(count, "cudaMalloc(x stage)");
                deviceStageY_[stage].allocate(count, "cudaMalloc(y stage)");
            }
            deviceStageXPtrs_.allocate(deviceStageX_.size(), "cudaMalloc(x-stage pointers)");
            deviceStageYPtrs_.allocate(deviceStageY_.size(), "cudaMalloc(y-stage pointers)");
            deviceBlockErrors_.allocate((count + threadsPerBlock - 1) / threadsPerBlock,
                                        "cudaMalloc(DOPRI5 errors)");
            deviceFailure_.allocate(1, "cudaMalloc(error flag)");
            refreshStagePointers();
            capacity_ = count;
        } catch (...) {
            release();
            throw;
        }
    }
    void release() const noexcept {
        deviceX_.reset();
        deviceY_.reset();
        deviceGamma_.reset();
        deviceU_.reset();
        deviceV_.reset();
        deviceInitialX_.reset();
        deviceInitialY_.reset();
        deviceTemporaryX_.reset();
        deviceTemporaryY_.reset();
        for (auto &stage : deviceStageX_)
            stage.reset();
        for (auto &stage : deviceStageY_)
            stage.reset();
        deviceStageXPtrs_.reset();
        deviceStageYPtrs_.reset();
        deviceBlockErrors_.reset();
        deviceFailure_.reset();
        capacity_ = 0;
        stateCount_ = 0;
        deviceStateValid_ = false;
        fsalValid_ = false;
    }
    SimParams params_;
    DeviceGeometry geometry_;
    std::unique_ptr<VelocityKernel> cpu_;
    mutable CudaBuffer<double> deviceX_, deviceY_, deviceGamma_;
    mutable CudaBuffer<double> deviceU_, deviceV_;
    mutable CudaBuffer<double> deviceInitialX_, deviceInitialY_;
    mutable CudaBuffer<double> deviceTemporaryX_, deviceTemporaryY_;
    mutable std::array<CudaBuffer<double>, 7> deviceStageX_;
    mutable std::array<CudaBuffer<double>, 7> deviceStageY_;
    mutable CudaBuffer<double *> deviceStageXPtrs_, deviceStageYPtrs_;
    mutable CudaBuffer<double> deviceBlockErrors_;
    mutable CudaBuffer<int> deviceFailure_;
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
