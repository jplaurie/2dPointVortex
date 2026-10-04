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

  private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};

struct DeviceGeometry {
    BoundaryKind boundary;
    float coreRadiusSquared;
    double boxLengthX;
    double boxLengthY;
    double diskRadius;
    float waveNumber;
    float periodicScale;
    float inverseRadius;
    int periodicImageLayers;
};

__device__ inline void setFailure(int *failure) { atomicExch(failure, 1); }

template <BoundaryKind boundary>
__device__ __forceinline__ void accumulateInteraction(
    const double *x, const double *y, const float *gamma, std::size_t target,
    std::size_t source, double targetX, double targetY, float normalizedTargetX,
    float normalizedTargetY, DeviceGeometry geometry, float &velocityX, float &velocityY,
    int *failure) {
    constexpr float inverseTwoPi = 0.15915494309189533577F;
    const float sourceGamma = gamma[source];

    if constexpr (boundary == BoundaryKind::infinite) {
        if (source == target)
            return;
        const float dx = static_cast<float>(targetX - x[source]);
        const float dy = static_cast<float>(targetY - y[source]);
        const float denominator = dx * dx + dy * dy + geometry.coreRadiusSquared;
        if (denominator == 0.0F || !isfinite(denominator)) {
            setFailure(failure);
            return;
        }
        const float coefficient = inverseTwoPi * sourceGamma / denominator;
        velocityX -= coefficient * dy;
        velocityY += coefficient * dx;
    } else if constexpr (boundary == BoundaryKind::periodic_x) {
        if (source == target)
            return;
        const float scaledX = static_cast<float>(
            static_cast<double>(geometry.waveNumber) *
            remainder(targetX - x[source], geometry.boxLengthX));
        const float scaledY = static_cast<float>(
            static_cast<double>(geometry.waveNumber) * (targetY - y[source]));
        float sinhRatio = 0.0F, sineRatio = 0.0F;
        if (fabsf(scaledY) <= 40.0F) {
            const float sinhHalfY = sinhf(0.5F * scaledY);
            const float sinHalfX = sinf(0.5F * scaledX);
            const float denominator =
                2.0F * (sinhHalfY * sinhHalfY + sinHalfX * sinHalfX);
            if (denominator == 0.0F || !isfinite(denominator)) {
                setFailure(failure);
                return;
            }
            sinhRatio = sinhf(scaledY) / denominator;
            sineRatio = sinf(scaledX) / denominator;
        } else {
            const float q = expf(-fabsf(scaledY));
            const float denominator = 1.0F + q * q - 2.0F * q * cosf(scaledX);
            sinhRatio = copysignf((1.0F - q * q) / denominator, scaledY);
            sineRatio = 2.0F * q * sinf(scaledX) / denominator;
        }
        velocityX -= geometry.periodicScale * sourceGamma * sinhRatio;
        velocityY += geometry.periodicScale * sourceGamma * sineRatio;
    } else if constexpr (boundary == BoundaryKind::periodic) {
        if (source == target)
            return;
        constexpr float twoPi = 6.28318530717958647693F;
        const float dx = static_cast<float>(
            static_cast<double>(geometry.waveNumber) *
            remainder(targetX - x[source], geometry.boxLengthX));
        const float dy = static_cast<float>(
            static_cast<double>(geometry.waveNumber) *
            remainder(targetY - y[source], geometry.boxLengthY));
        const float sineX = sinf(dx), sineY = sinf(dy);
        const float sinHalfX = sinf(0.5F * dx), sinHalfY = sinf(0.5F * dy);
        for (int image = -geometry.periodicImageLayers;
             image <= geometry.periodicImageLayers; ++image) {
            const float shiftedX = dx - twoPi * static_cast<float>(image);
            const float shiftedY = dy - twoPi * static_cast<float>(image);
            const float sinhHalfX =
                fabsf(shiftedX) > 40.0F ? CUDART_INF_F : sinhf(0.5F * shiftedX);
            const float sinhHalfY =
                fabsf(shiftedY) > 40.0F ? CUDART_INF_F : sinhf(0.5F * shiftedY);
            const float denominatorU =
                2.0F * (sinhHalfX * sinhHalfX + sinHalfY * sinHalfY);
            const float denominatorV =
                2.0F * (sinhHalfY * sinhHalfY + sinHalfX * sinHalfX);
            if (denominatorU == 0.0F || denominatorV == 0.0F) {
                setFailure(failure);
                continue;
            }
            velocityX -= geometry.periodicScale * sourceGamma * sineY / denominatorU;
            velocityY += geometry.periodicScale * sourceGamma * sineX / denominatorV;
        }
    } else {
        if (source != target) {
            const float dx = static_cast<float>(targetX - x[source]);
            const float dy = static_cast<float>(targetY - y[source]);
            const float denominator = dx * dx + dy * dy;
            if (denominator == 0.0F || !isfinite(denominator)) {
                setFailure(failure);
            } else {
                const float coefficient = inverseTwoPi * sourceGamma / denominator;
                velocityX -= coefficient * dy;
                velocityY += coefficient * dx;
            }
        }
        const float sourceX = static_cast<float>(
            x[source] * static_cast<double>(geometry.inverseRadius));
        const float sourceY = static_cast<float>(
            y[source] * static_cast<double>(geometry.inverseRadius));
        const float a =
            1.0F - (normalizedTargetX * sourceX + normalizedTargetY * sourceY);
        const float b = normalizedTargetY * sourceX - normalizedTargetX * sourceY;
        const float denominator = a * a + b * b;
        if (denominator == 0.0F || !isfinite(denominator)) {
            setFailure(failure);
            return;
        }
        const float imageX = -a * sourceX - b * sourceY;
        const float imageY = -a * sourceY + b * sourceX;
        const float coefficient =
            -inverseTwoPi * sourceGamma * geometry.inverseRadius / denominator;
        velocityX -= coefficient * imageY;
        velocityY += coefficient * imageX;
    }
}

template <BoundaryKind boundary>
__global__ void velocityKernel(const double *x, const double *y, const float *gamma, double *u,
                               double *v, std::size_t count, std::size_t begin, std::size_t end,
                               DeviceGeometry geometry, int *failure) {
    const std::size_t target =
        begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (target >= end)
        return;
    const double targetX = x[target], targetY = y[target];
    if (!isfinite(targetX) || !isfinite(targetY) ||
        (boundary == BoundaryKind::disk &&
         targetX * targetX + targetY * targetY >= geometry.diskRadius * geometry.diskRadius)) {
        setFailure(failure);
        return;
    }

    const float normalizedTargetX = static_cast<float>(
        targetX * static_cast<double>(geometry.inverseRadius));
    const float normalizedTargetY = static_cast<float>(
        targetY * static_cast<double>(geometry.inverseRadius));
    float ux0 = 0.0F, ux1 = 0.0F, ux2 = 0.0F, ux3 = 0.0F;
    float uy0 = 0.0F, uy1 = 0.0F, uy2 = 0.0F, uy3 = 0.0F;
    std::size_t source = 0;
    for (; source + 3 < count; source += 4) {
        accumulateInteraction<boundary>(x, y, gamma, target, source, targetX, targetY,
                                        normalizedTargetX, normalizedTargetY, geometry, ux0, uy0,
                                        failure);
        accumulateInteraction<boundary>(x, y, gamma, target, source + 1, targetX, targetY,
                                        normalizedTargetX, normalizedTargetY, geometry, ux1, uy1,
                                        failure);
        accumulateInteraction<boundary>(x, y, gamma, target, source + 2, targetX, targetY,
                                        normalizedTargetX, normalizedTargetY, geometry, ux2, uy2,
                                        failure);
        accumulateInteraction<boundary>(x, y, gamma, target, source + 3, targetX, targetY,
                                        normalizedTargetX, normalizedTargetY, geometry, ux3, uy3,
                                        failure);
    }
    for (; source < count; ++source)
        accumulateInteraction<boundary>(x, y, gamma, target, source, targetX, targetY,
                                        normalizedTargetX, normalizedTargetY, geometry, ux0, uy0,
                                        failure);
    const float velocityX = (ux0 + ux1) + (ux2 + ux3);
    const float velocityY = (uy0 + uy1) + (uy2 + uy3);
    if (!isfinite(velocityX) || !isfinite(velocityY))
        setFailure(failure);
    else {
        u[target] = static_cast<double>(velocityX);
        v[target] = static_cast<double>(velocityY);
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
        setFailure(failure);
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
        if (coefficients[stage] != 0.0) {
            dx += coefficients[stage] * stageX[stage][index];
            dy += coefficients[stage] * stageY[stage][index];
        }
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
        const double scaleX = absoluteTolerance + relativeTolerance *
                                                      fmax(fabs(initialX[index]),
                                                           fabs(candidateX[index]));
        const double scaleY = absoluteTolerance + relativeTolerance *
                                                      fmax(fabs(initialY[index]),
                                                           fabs(candidateY[index]));
        if (!isfinite(errorX) || !isfinite(errorY) || !isfinite(scaleX) ||
            !isfinite(scaleY) || !(scaleX > 0.0) || !(scaleY > 0.0))
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

class CudaMixedKernel final : public VelocityKernel, public DeviceStepper {
  public:
    explicit CudaMixedKernel(const SimParams &params)
        : params_(params), cpu_(makeReferenceKernel(params)) {
        geometry_.boundary = params.boundary;
        geometry_.coreRadiusSquared = narrowFloat(params.coreRadius * params.coreRadius,
                                                   "core-radius squared", true);
        geometry_.boxLengthX = params.boxLengthX;
        geometry_.boxLengthY = params.boxLengthY;
        geometry_.diskRadius = params.diskRadius;
        geometry_.periodicImageLayers = params.periodicImageLayers;
        if (params.boundary == BoundaryKind::periodic_x ||
            params.boundary == BoundaryKind::periodic) {
            geometry_.waveNumber =
                narrowFloat(2.0 * CUDART_PI / params.boxLengthX, "periodic wave number", false);
            geometry_.periodicScale =
                narrowFloat(0.5 / params.boxLengthX, "periodic velocity scale", false);
        }
        if (params.boundary == BoundaryKind::disk)
            geometry_.inverseRadius =
                narrowFloat(1.0 / params.diskRadius, "inverse disk radius", false);
    }

    void evaluateRange(const std::vector<double> &x, const std::vector<double> &y,
                       const std::vector<double> &gamma, VelocityField &velocity,
                       std::size_t begin, std::size_t end) const override {
        validateVortexArrays(x, y, gamma);
        const std::size_t count = x.size();
        if (begin > end || end > count)
            throw std::invalid_argument("invalid CUDA mixed vortex arrays or target range");
        validateGeometry(x, y, gamma);
        velocity.resize(count);
        if (begin == end)
            return;
        ensureCapacity(count);
        stateCount_ = count;
        deviceStateValid_ = true;
        fsalValid_ = false;
        const std::size_t bytes = count * sizeof(double);
        cudaCheck(cudaMemcpy(deviceX_.get(), x.data(), bytes, cudaMemcpyHostToDevice),
                  "copy x to mixed GPU backend");
        cudaCheck(cudaMemcpy(deviceY_.get(), y.data(), bytes, cudaMemcpyHostToDevice),
                  "copy y to mixed GPU backend");
        uploadGamma(gamma);
        evaluateDevice(deviceX_.get(), deviceY_.get(), deviceU_.get(), deviceV_.get(), begin, end);
        cudaCheck(cudaMemcpy(velocity.x.data() + begin, deviceU_.get() + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy mixed u from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data() + begin, deviceV_.get() + begin,
                             (end - begin) * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy mixed v from GPU");
    }

    double hamiltonian(const VortexSystem &state) const override { return cpu_->hamiltonian(state); }

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
                  "upload mixed state x to GPU");
        cudaCheck(cudaMemcpy(deviceY_.get(), state.y.data(), bytes, cudaMemcpyHostToDevice),
                  "upload mixed state y to GPU");
        uploadGamma(state.circulation);
    }

    void downloadState(VortexSystem &state) const override {
        requireDeviceState();
        if (state.size() != stateCount_)
            throw std::runtime_error("host and CUDA mixed vortex populations differ");
        if (stateCount_ == 0)
            return;
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(state.x.data(), deviceX_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download mixed state x from GPU");
        cudaCheck(cudaMemcpy(state.y.data(), deviceY_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download mixed state y from GPU");
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
                  "download mixed velocity x from GPU");
        cudaCheck(cudaMemcpy(velocity.y.data(), deviceV_.get(), bytes, cudaMemcpyDeviceToHost),
                  "download mixed velocity y from GPU");
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
        rk4CombineKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            deviceX_.get(), deviceY_.get(), deviceInitialX_.get(), deviceInitialY_.get(),
            deviceStageXPtrs_.get(), deviceStageYPtrs_.get(), stateCount_, dt);
        cudaCheck(cudaGetLastError(), "launch CUDA mixed RK4 final stage");
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
                          "accept CUDA mixed DOPRI5 x state");
                cudaCheck(cudaMemcpy(deviceY_.get(), deviceTemporaryY_.get(), bytes,
                                     cudaMemcpyDeviceToDevice),
                          "accept CUDA mixed DOPRI5 y state");
                std::swap(deviceStageX_[0], deviceStageX_[6]);
                std::swap(deviceStageY_[0], deviceStageY_[6]);
                refreshStagePointers();
                fsalValid_ = true;
                validateDeviceState(deviceX_.get(), deviceY_.get());
                return {dt, suggested, error, rejected};
            }
            if (dt <= minimumTimeStep || ++rejected > 32)
                throw std::runtime_error(
                    "adaptive CUDA mixed integrator could not satisfy tolerance");
            dt = std::max(minimumTimeStep, std::min(suggested, dt * 0.9));
        }
    }

    void invalidateDerivative() const noexcept override { fsalValid_ = false; }

  private:
    static constexpr int threadsPerBlock = 256;

    static float narrowFloat(double value, const char *quantity, bool allowZero) {
        const float result = static_cast<float>(value);
        if (!std::isfinite(value) || !std::isfinite(result) ||
            (allowZero ? result < 0.0F : result <= 0.0F) ||
            (value > 0.0 && result == 0.0F))
            throw std::invalid_argument(std::string("CUDA mixed backend cannot represent ") +
                                        quantity + " in FP32");
        return result;
    }

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

    void uploadGamma(const std::vector<double> &gamma) const {
        hostGammaFloat_.resize(gamma.size());
        for (std::size_t index = 0; index < gamma.size(); ++index) {
            hostGammaFloat_[index] = static_cast<float>(gamma[index]);
            if (!std::isfinite(hostGammaFloat_[index]) ||
                (gamma[index] != 0.0 && hostGammaFloat_[index] == 0.0F))
                throw std::invalid_argument(
                    "CUDA mixed backend cannot represent circulation in FP32");
        }
        cudaCheck(cudaMemcpy(deviceGamma_.get(), hostGammaFloat_.data(),
                             gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
                  "upload mixed circulation to GPU");
    }

    void requireDeviceState() const {
        if (!deviceStateValid_)
            throw std::logic_error("CUDA mixed device state has not been uploaded");
    }

    void refreshStagePointers() const {
        std::array<double *, 7> stageXPointers{}, stageYPointers{};
        for (std::size_t stage = 0; stage < deviceStageX_.size(); ++stage) {
            stageXPointers[stage] = deviceStageX_[stage].get();
            stageYPointers[stage] = deviceStageY_[stage].get();
        }
        cudaCheck(cudaMemcpy(deviceStageXPtrs_.get(), stageXPointers.data(),
                             stageXPointers.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA mixed x-stage pointers");
        cudaCheck(cudaMemcpy(deviceStageYPtrs_.get(), stageYPointers.data(),
                             stageYPointers.size() * sizeof(double *), cudaMemcpyHostToDevice),
                  "upload CUDA mixed y-stage pointers");
    }

    void checkFailure(const char *operation) const {
        int failure = 0;
        cudaCheck(cudaMemcpy(&failure, deviceFailure_.get(), sizeof(int), cudaMemcpyDeviceToHost),
                  "read CUDA mixed validation flag");
        if (failure)
            throw std::runtime_error(std::string(operation) +
                                     ": non-finite, coincident, or out-of-domain vortex state");
    }

    void validateDeviceState(const double *x, const double *y) const {
        cudaCheck(cudaMemset(deviceFailure_.get(), 0, sizeof(int)),
                  "clear CUDA mixed state-validation flag");
        validateStateKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            x, y, stateCount_, geometry_.boundary, geometry_.diskRadius * geometry_.diskRadius,
            deviceFailure_.get());
        cudaCheck(cudaGetLastError(), "launch CUDA mixed state-validation kernel");
        checkFailure("CUDA mixed state validation failed");
    }

    template <BoundaryKind boundary>
    void launchVelocity(const double *x, const double *y, double *u, double *v, std::size_t begin,
                        std::size_t end) const {
        velocityKernel<boundary><<<blockCount(end - begin), threadsPerBlock>>>(
            x, y, deviceGamma_.get(), u, v, stateCount_, begin, end, geometry_,
            deviceFailure_.get());
    }

    void evaluateDevice(const double *x, const double *y, double *u, double *v, std::size_t begin,
                        std::size_t end) const {
        if (begin == end)
            return;
        cudaCheck(cudaMemset(deviceFailure_.get(), 0, sizeof(int)),
                  "clear CUDA mixed velocity error flag");
        switch (geometry_.boundary) {
        case BoundaryKind::infinite:
            launchVelocity<BoundaryKind::infinite>(x, y, u, v, begin, end);
            break;
        case BoundaryKind::periodic_x:
            launchVelocity<BoundaryKind::periodic_x>(x, y, u, v, begin, end);
            break;
        case BoundaryKind::periodic:
            launchVelocity<BoundaryKind::periodic>(x, y, u, v, begin, end);
            break;
        case BoundaryKind::disk:
            launchVelocity<BoundaryKind::disk>(x, y, u, v, begin, end);
            break;
        }
        cudaCheck(cudaGetLastError(), "launch CUDA mixed velocity kernel");
        checkFailure("CUDA mixed velocity evaluation failed");
    }

    void copyStateToInitial() const {
        const std::size_t bytes = stateCount_ * sizeof(double);
        cudaCheck(cudaMemcpy(deviceInitialX_.get(), deviceX_.get(), bytes,
                             cudaMemcpyDeviceToDevice),
                  "copy CUDA mixed initial x state");
        cudaCheck(cudaMemcpy(deviceInitialY_.get(), deviceY_.get(), bytes,
                             cudaMemcpyDeviceToDevice),
                  "copy CUDA mixed initial y state");
    }

    void makeStage(double dt, const std::array<double, 7> &coefficients) const {
        makeStageKernel<<<blockCount(stateCount_), threadsPerBlock>>>(
            deviceTemporaryX_.get(), deviceTemporaryY_.get(), deviceInitialX_.get(),
            deviceInitialY_.get(), deviceStageXPtrs_.get(), deviceStageYPtrs_.get(), stateCount_,
            dt, coefficients[0], coefficients[1], coefficients[2], coefficients[3],
            coefficients[4], coefficients[5], coefficients[6]);
        cudaCheck(cudaGetLastError(), "launch CUDA mixed Runge--Kutta stage");
    }

    void makeDopriStages(double dt) const {
        for (std::size_t stage = 1; stage < 7; ++stage) {
            makeStage(dt, integrator_detail::dopriCoefficients[stage]);
            evaluateDevice(deviceTemporaryX_.get(), deviceTemporaryY_.get(),
                           deviceStageX_[stage].get(), deviceStageY_[stage].get(), 0,
                           stateCount_);
        }
    }

    double dopriError(double dt, double absoluteTolerance, double relativeTolerance) const {
        const int blocks = blockCount(stateCount_);
        dopriErrorKernel<<<blocks, threadsPerBlock>>>(
            deviceInitialX_.get(), deviceInitialY_.get(), deviceTemporaryX_.get(),
            deviceTemporaryY_.get(), deviceStageXPtrs_.get(), deviceStageYPtrs_.get(),
            stateCount_, dt, absoluteTolerance, relativeTolerance, deviceBlockErrors_.get());
        cudaCheck(cudaGetLastError(), "launch CUDA mixed DOPRI5 error kernel");
        hostBlockErrors_.resize(static_cast<std::size_t>(blocks));
        cudaCheck(cudaMemcpy(hostBlockErrors_.data(), deviceBlockErrors_.get(),
                             hostBlockErrors_.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "download CUDA mixed DOPRI5 error blocks");
        return *std::max_element(hostBlockErrors_.begin(), hostBlockErrors_.end());
    }

    void ensureCapacity(std::size_t count) const {
        if (count <= capacity_)
            return;
        release();
        try {
            deviceX_.allocate(count, "cudaMalloc(mixed x)");
            deviceY_.allocate(count, "cudaMalloc(mixed y)");
            deviceGamma_.allocate(count, "cudaMalloc(mixed circulation)");
            deviceU_.allocate(count, "cudaMalloc(mixed u)");
            deviceV_.allocate(count, "cudaMalloc(mixed v)");
            deviceInitialX_.allocate(count, "cudaMalloc(mixed initial x)");
            deviceInitialY_.allocate(count, "cudaMalloc(mixed initial y)");
            deviceTemporaryX_.allocate(count, "cudaMalloc(mixed temporary x)");
            deviceTemporaryY_.allocate(count, "cudaMalloc(mixed temporary y)");
            for (std::size_t stage = 0; stage < deviceStageX_.size(); ++stage) {
                deviceStageX_[stage].allocate(count, "cudaMalloc(mixed x stage)");
                deviceStageY_[stage].allocate(count, "cudaMalloc(mixed y stage)");
            }
            deviceStageXPtrs_.allocate(deviceStageX_.size(),
                                       "cudaMalloc(mixed x-stage pointers)");
            deviceStageYPtrs_.allocate(deviceStageY_.size(),
                                       "cudaMalloc(mixed y-stage pointers)");
            deviceBlockErrors_.allocate((count + threadsPerBlock - 1) / threadsPerBlock,
                                        "cudaMalloc(mixed DOPRI5 errors)");
            deviceFailure_.allocate(1, "cudaMalloc(mixed error flag)");
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
    DeviceGeometry geometry_{};
    std::unique_ptr<VelocityKernel> cpu_;
    mutable CudaBuffer<double> deviceX_, deviceY_;
    mutable CudaBuffer<float> deviceGamma_;
    mutable CudaBuffer<double> deviceU_, deviceV_;
    mutable CudaBuffer<double> deviceInitialX_, deviceInitialY_;
    mutable CudaBuffer<double> deviceTemporaryX_, deviceTemporaryY_;
    mutable std::array<CudaBuffer<double>, 7> deviceStageX_, deviceStageY_;
    mutable CudaBuffer<double *> deviceStageXPtrs_, deviceStageYPtrs_;
    mutable CudaBuffer<double> deviceBlockErrors_;
    mutable CudaBuffer<int> deviceFailure_;
    mutable std::vector<float> hostGammaFloat_;
    mutable std::vector<double> hostBlockErrors_;
    mutable std::size_t capacity_ = 0;
    mutable std::size_t stateCount_ = 0;
    mutable bool deviceStateValid_ = false;
    mutable bool fsalValid_ = false;
};

} // namespace

void backendInitialize(int &, char **&) { cudaCheck(cudaFree(nullptr), "initialize CUDA mixed"); }
void backendFinalize() {}
void backendAbort(int) {}
bool backendIsRoot() { return true; }
const char *backendName() { return "CUDA-mixed-FP32/FP64"; }
std::string backendRuntimeDetails() {
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "query CUDA mixed device");
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, device), "query CUDA mixed device properties");
    return "cuda_device " + std::to_string(device) + "\ncuda_device_name \"" +
           std::string(properties.name) + "\"\ncuda_compute_capability " +
           std::to_string(properties.major) + "." + std::to_string(properties.minor) +
           "\ncuda_precision mixed\ncuda_interactions fp32\ncuda_state fp64" +
           "\ncuda_integrators rk4,dopri5" +
           "\ncuda_boundaries infinite,periodic_x,periodic,disk";
}
std::unique_ptr<VelocityKernel> makeBackendKernel(const SimParams &params) {
    return std::make_unique<CudaMixedKernel>(params);
}
