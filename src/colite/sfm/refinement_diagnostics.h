#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "colite/core/geometry.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::refinement_internal {
using ObservationKey = std::pair<ImageId, Point2DIdx>;
struct ObservationState {
  Point3DId point;
  double error;
};
using Snapshot = std::map<ObservationKey, ObservationState>;

inline Snapshot Capture(const Reconstruction& model) {
  Snapshot result;
  for (const auto& [id, point] : model.Points3D()) {
    for (const auto& element : point.track.elements) {
      if (!model.IsImageRegistered(element.image_id)) continue;
      const auto& image = model.GetImage(element.image_id);
      if (!image.HasPose() || element.point2d_idx >= image.points2d.size()) continue;
      const auto error = ReprojectionError(model.GetCamera(image.camera_id),
          *image.cam_from_world, point.xyz, image.points2d[element.point2d_idx].xy);
      result.emplace(ObservationKey{element.image_id, element.point2d_idx},
                     ObservationState{id, error.value_or(-1.0)});
    }
  }
  return result;
}

inline double P95(std::vector<double> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
      [](double value) { return !std::isfinite(value) || value < 0; }), values.end());
  if (values.empty()) return -1.0;
  std::sort(values.begin(), values.end());
  const double index = 0.95 * (values.size() - 1);
  const auto low = static_cast<std::size_t>(index);
  const auto high = std::min(low + 1, values.size() - 1);
  return values[low] + (values[high] - values[low]) * (index - low);
}

inline double P95(const Snapshot& state) {
  std::vector<double> values;
  values.reserve(state.size());
  for (const auto& [key, observation] : state) values.push_back(observation.error);
  return P95(std::move(values));
}

inline void Record(const Snapshot& before, const Snapshot& after_ba,
                   const Snapshot& completed, const Snapshot& filtered,
                   std::set<ObservationKey>* previously_removed,
                   GlobalRefinementDiagnostics* row,
                   std::vector<GlobalRefinementObservationChange>* events,
                   double max_error = 4.0) {
  row->traced = true;
  row->p95_before_ba = P95(before);
  row->p95_after_ba = P95(after_ba);
  row->p95_after_completion = P95(completed);
  row->p95_after_filter = P95(filtered);
  std::vector<double> common_before, common_after;
  for (const auto& [key, observation] : after_ba) {
    if (!completed.count(key)) {
      ++row->unique_removed;
      ++row->completion_removed;
      events->push_back({row->iteration, "completion_remove", key.first, key.second,
          observation.point, kInvalidPoint3DId, observation.error, -1.0,
          previously_removed->count(key) > 0, "merge_pruning"});
      previously_removed->insert(key);
    }
  }
  for (const auto& [key, observation] : completed) {
    const auto old = after_ba.find(key);
    const bool added = old == after_ba.end();
    const bool reassigned = !added && old->second.point != observation.point;
    const bool repeated = previously_removed->count(key) > 0;
    if (added) {
      ++row->unique_added;
      if (repeated) ++row->readded_previous;
    }
    if (reassigned) ++row->reassigned;
    if (added || reassigned) {
      events->push_back({row->iteration, added ? "completion_add" : "completion_reassign",
          key.first, key.second, added ? kInvalidPoint3DId : old->second.point,
          observation.point, added ? -1.0 : old->second.error, observation.error, repeated});
    }
    if (!filtered.count(key)) {
      ++row->unique_removed;
      if (added) ++row->added_then_removed;
      events->push_back({row->iteration, "filter_remove", key.first, key.second,
          observation.point, kInvalidPoint3DId, observation.error, -1.0, repeated});
      events->back().reason = !std::isfinite(observation.error) || observation.error < 0
          ? "invalid_projection" : observation.error > max_error
          ? "reprojection" : "point_filter";
      previously_removed->insert(key);
    }
  }
  for (const auto& [key, observation] : before) {
    const auto after = filtered.find(key);
    if (after == filtered.end()) ++row->net_membership_changes;
    else {
      common_before.push_back(observation.error);
      common_after.push_back(after->second.error);
    }
  }
  for (const auto& [key, observation] : filtered) {
    if (!before.count(key)) ++row->net_membership_changes;
  }
  row->common_p95_before = P95(std::move(common_before));
  row->common_p95_after = P95(std::move(common_after));
}
}  // namespace colite::refinement_internal
