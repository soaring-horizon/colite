#pragma once

#include <cstddef>

#include <Eigen/Core>
#include <ceres/cost_function.h>

#include "colite/core/camera.h"

namespace colite::bundle_internal {

// Parameter blocks: Eigen quaternion XYZW, translation, world point, camera.
// The caller transfers ownership of the returned cost to ceres::Problem.
ceres::CostFunction* CreateReprojectionCost(
    CameraModelId model_id, std::size_t num_camera_params,
    const Eigen::Vector2d& observed_xy);

}  // namespace colite::bundle_internal
