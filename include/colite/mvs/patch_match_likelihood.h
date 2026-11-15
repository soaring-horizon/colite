#pragma once

#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define COLITE_MVS_HD __host__ __device__
#else
#define COLITE_MVS_HD
#endif

namespace colite::mvs::detail {

// COLMAP increases history influence from zero to (N-1)/N over N sweeps.
inline float SelectionHistoryWeight(std::size_t step, std::size_t num_steps) {
  if (num_steps == 0) return 0.0F;
  return static_cast<float>(step < num_steps ? step : num_steps - 1) /
         static_cast<float>(num_steps);
}

// COLMAP cost convention: cost = 1 - NCC, in [0, 2].
COLITE_MVS_HD inline float NccLikelihood(float cost, float sigma) {
  constexpr float pi = 3.14159265358979323846F;
  const float norm = 2.0F / (::sqrtf(2.0F * pi) * sigma *
                             ::erff(2.0F / (sigma * ::sqrtf(2.0F))));
  return ::expf(-0.5F * cost * cost / (sigma * sigma)) * norm;
}

COLITE_MVS_HD inline float HmmForward(float emission, float previous) {
  constexpr float stay = 0.99999F;
  constexpr float change = 1.0F - stay;
  emission = ::fabsf(emission);  // Negative marks an invalid photometric sample.
  const float no = (previous * change + (1.0F - previous) * stay) * 0.5F;
  const float yes = (previous * stay + (1.0F - previous) * change) * emission;
  return no + yes > 1e-12F ? yes / (no + yes) : 0.5F;
}

COLITE_MVS_HD inline float HmmBackward(float emission, float previous) {
  constexpr float stay = 0.99999F;
  constexpr float change = 1.0F - stay;
  emission = ::fabsf(emission);
  const float no = previous * emission * change + (1.0F - previous) * 0.5F * stay;
  const float yes = previous * emission * stay + (1.0F - previous) * 0.5F * change;
  return no + yes > 1e-12F ? yes / (no + yes) : 0.5F;
}

COLITE_MVS_HD inline float HmmSelection(float alpha, float beta,
                                      float previous, float previous_weight) {
  const float no = (1.0F - alpha) * (1.0F - beta);
  const float yes = alpha * beta;
  const float current = no + yes > 1e-12F ? yes / (no + yes) : 0.5F;
  return previous_weight * previous + (1.0F - previous_weight) * current;
}

}  // namespace colite::mvs::detail

#undef COLITE_MVS_HD
