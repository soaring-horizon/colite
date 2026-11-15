#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/core/parallel.h"

namespace colite::bundle_internal {

inline ceres::Solver::Options CreateSolverOptions(
    const BundleAdjustmentSolverOptions& controls, int max_num_iterations,
    std::size_t num_images, const ceres::Problem& problem) {
  ceres::Solver::Options solver;
  solver.max_num_iterations = max_num_iterations;
  solver.max_linear_solver_iterations = controls.max_linear_solver_iterations;
  solver.function_tolerance = controls.function_tolerance;
  solver.gradient_tolerance = controls.gradient_tolerance;
  solver.parameter_tolerance = controls.parameter_tolerance;
  solver.max_num_consecutive_invalid_steps = 10;
  solver.max_consecutive_nonmonotonic_steps = 10;
  solver.logging_type = ceres::SILENT;
  solver.minimizer_progress_to_stdout = false;
  if (num_images <= static_cast<std::size_t>(controls.max_num_images_direct_dense_cpu_solver)) {
    solver.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (solver.sparse_linear_algebra_library_type != ceres::NO_SPARSE &&
             num_images <= static_cast<std::size_t>(controls.max_num_images_direct_sparse_cpu_solver)) {
    solver.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {
    solver.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver.preconditioner_type = ceres::SCHUR_JACOBI;
  }
  solver.num_threads = problem.NumResiduals() < controls.min_num_residuals_for_cpu_multi_threading
      ? 1 : parallel::getNumThreads();
  return solver;
}

// A fixed pose removes six similarity degrees of freedom, but leaves scale.
// Choose a translation coordinate with a nonzero baseline, including when
// the caller already fixed one pose or the first two camera centers coincide.
template <typename PoseBlock>
std::map<ImageId, std::vector<int>> GaugeTranslationConstraints(
    BundleGaugeFixing fixing,
    const std::map<ImageId, PoseBlock>& poses,
    const std::set<ImageId>& constant_poses,
    const std::map<ImageId, std::vector<int>>& explicit_constraints,
    std::set<ImageId>* added_constant_poses) {
  auto constraints = explicit_constraints;
  if (fixing == BundleGaugeFixing::kNone || poses.empty()) return constraints;
  auto anchor = poses.begin();
  bool has_fixed_anchor = false;
  for (const auto id : constant_poses) {
    const auto it = poses.find(id);
    if (it != poses.end()) {
      anchor = it;
      has_fixed_anchor = true;
      break;
    }
  }
  if (!has_fixed_anchor) added_constant_poses->insert(anchor->first);
  if (fixing == BundleGaugeFixing::kFirstImage) return constraints;
  const auto center = [](const PoseBlock& block) -> Eigen::Vector3d {
    const Eigen::Quaterniond q(block.q_xyzw[3], block.q_xyzw[0],
                               block.q_xyzw[1], block.q_xyzw[2]);
    return q.normalized().conjugate() *
           (-Eigen::Vector3d(block.t[0], block.t[1], block.t[2]));
  };
  const Eigen::Vector3d anchor_center = center(anchor->second);
  // Two distinct fixed camera centers already constrain scale.
  for (const auto id : constant_poses) {
    const auto it = poses.find(id);
    if (it != poses.end() &&
        (center(it->second) - anchor_center).norm() > 1e-9) {
      return constraints;
    }
  }
  for (const auto& [id, block] : poses) {
    if (id == anchor->first || constant_poses.count(id)) continue;
    const Eigen::Quaterniond q(block.q_xyzw[3], block.q_xyzw[0],
                               block.q_xyzw[1], block.q_xyzw[2]);
    const Eigen::Vector3d baseline =
        q.normalized() * anchor_center +
        Eigen::Vector3d(block.t[0], block.t[1], block.t[2]);
    Eigen::Index axis;
    if (baseline.cwiseAbs().maxCoeff(&axis) <= 1e-9) continue;
    auto& indices = constraints[id];
    if (std::none_of(indices.begin(), indices.end(), [&](int index) {
          return index >= 0 && index < 3 && std::abs(baseline[index]) > 1e-9;
        })) {
      indices.push_back(static_cast<int>(axis));
    }
    break;
  }
  return constraints;
}

template <typename Summary>
void RecordSolverSummary(const ceres::Solver::Summary& solver, Summary* summary) {
  summary->linear_solver_type = ceres::LinearSolverTypeToString(solver.linear_solver_type_used);
  summary->solver_num_threads = solver.num_threads_used;
  summary->initial_cost = solver.initial_cost;
  summary->final_cost = solver.final_cost;
  summary->num_unsuccessful_steps = solver.num_unsuccessful_steps;
  for (const auto& iteration : solver.iterations) {
    if (iteration.iteration > 0 && !iteration.step_is_valid) {
      ++summary->num_invalid_steps;
    }
  }
  summary->message = solver.BriefReport();
}

template <typename PoseBlocks, typename PointBlocks, typename CameraBlocks>
bool CanCommitSolution(const ceres::Solver::Summary& solver,
                       const PoseBlocks& poses,
                       const PointBlocks& points,
                       const CameraBlocks& cameras) {
  if (!solver.IsSolutionUsable() || !std::isfinite(solver.initial_cost) ||
      !std::isfinite(solver.final_cost) || solver.final_cost < 0 ||
      solver.final_cost > solver.initial_cost +
                              1e-10 * std::max(1.0, solver.initial_cost)) {
    return false;
  }
  for (const auto& [id, block] : poses) {
    const Eigen::Map<const Eigen::Vector4d> q(block.q_xyzw);
    if (!q.allFinite() || q.norm() <= 0 ||
        !Eigen::Map<const Eigen::Vector3d>(block.t).allFinite()) return false;
  }
  for (const auto& [id, block] : points) {
    if (!Eigen::Map<const Eigen::Vector3d>(block.xyz).allFinite()) return false;
  }
  for (const auto& [id, values] : cameras) {
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) return false;
  }
  return true;
}

}  // namespace colite::bundle_internal
