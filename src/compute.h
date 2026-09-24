#ifndef POINT_VORTEX_COMPUTE_H
#define POINT_VORTEX_COMPUTE_H
#include "vortex.h"
#include <array>
#include <vector>

void validatePeriodicCirculation(const std::vector<double> &circulation);
void validateDiskPositions(const std::vector<double> &x, const std::vector<double> &y,
                           double radiusSquared);

struct StepResult {
    double acceptedTimeStep = 0.0;
    double suggestedTimeStep = 0.0;
    double normalizedError = 0.0;
    unsigned rejectedSteps = 0;
};
// Geometry-independent right-hand side used by both time integrators.
class VelocityKernel {
  public:
    virtual ~VelocityKernel() = default;
    void evaluate(const VortexSystem &, VelocityField &) const;
    void evaluate(const std::vector<double> &x, const std::vector<double> &y,
                  const std::vector<double> &circulation, VelocityField &velocity) const;
    virtual void evaluateRange(const std::vector<double> &x, const std::vector<double> &y,
                               const std::vector<double> &circulation, VelocityField &velocity,
                               std::size_t begin, std::size_t end) const = 0;
    virtual double hamiltonian(const VortexSystem &) const = 0;
    // CUDA overrides these hooks to retain state and Runge--Kutta stages on the device.
    virtual bool supportsDeviceStepping() const noexcept { return false; }
    virtual void uploadDeviceState(const VortexSystem &) const;
    virtual void downloadDeviceState(VortexSystem &) const;
    virtual void evaluateDeviceState(VelocityField &) const;
    virtual void deviceRk4Step(double) const;
    virtual StepResult deviceDopri5Step(double, double, double, double, double) const;
    virtual void invalidateDeviceDerivative() const noexcept {}
};
class InfinitePlaneKernel final : public VelocityKernel {
  public:
    using VelocityKernel::evaluate;
    explicit InfinitePlaneKernel(double coreRadius = 0.0);
    void evaluateRange(const std::vector<double> &, const std::vector<double> &,
                       const std::vector<double> &, VelocityField &, std::size_t,
                       std::size_t) const override;
    double hamiltonian(const VortexSystem &) const override;

  private:
    double coreRadiusSquared_;
};
// Cylinder: periodic in x with period L and unbounded in y.
class PeriodicXKernel final : public VelocityKernel {
  public:
    using VelocityKernel::evaluate;
    explicit PeriodicXKernel(double lengthX);
    void evaluateRange(const std::vector<double> &, const std::vector<double> &,
                       const std::vector<double> &, VelocityField &, std::size_t,
                       std::size_t) const override;
    double hamiltonian(const VortexSystem &) const override;

  private:
    double lengthX_;
};
// Weiss--McWilliams rapidly convergent image sum for a square torus.
class PeriodicBoxKernel final : public VelocityKernel {
  public:
    using VelocityKernel::evaluate;
    PeriodicBoxKernel(double lengthX, double lengthY, int imageLayers = 8);
    void evaluateRange(const std::vector<double> &, const std::vector<double> &,
                       const std::vector<double> &, VelocityField &, std::size_t,
                       std::size_t) const override;
    double hamiltonian(const VortexSystem &) const override;

  private:
    double lengthX_;
    double lengthY_;
    int imageLayers_;
};
// Impermeable circular wall represented by opposite-sign inverse-point images.
class DiskKernel final : public VelocityKernel {
  public:
    using VelocityKernel::evaluate;
    explicit DiskKernel(double radius);
    void evaluateRange(const std::vector<double> &, const std::vector<double> &,
                       const std::vector<double> &, VelocityField &, std::size_t,
                       std::size_t) const override;
    double hamiltonian(const VortexSystem &) const override;

  private:
    double radius_;
    double radiusSquared_;
};
struct Invariants {
    double circulation = 0.0, linearImpulseX = 0.0, linearImpulseY = 0.0;
    double angularImpulse = 0.0, hamiltonian = 0.0;
};
Invariants computeInvariants(const VortexSystem &, double coreRadius = 0.0);
Invariants computeInvariants(const VortexSystem &, const VelocityKernel &kernel);
#endif
