#include "colite/matching/sift_matching_pipeline.h"
#include "colite/core/progress.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

#include "colite/core/parallel.h"

namespace colite {
namespace {

using Clock = std::chrono::steady_clock;
double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

class PairQueue {
 public:
  void Push(std::size_t value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) return;
      values_.push(value);
    }
    condition_.notify_one();
  }
  bool Pop(std::size_t* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !values_.empty(); });
    if (values_.empty()) return false;
    *value = values_.front();
    values_.pop();
    return true;
  }
  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    condition_.notify_all();
  }
 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::queue<std::size_t> values_;
  bool closed_ = false;
};

bool Usable(const TwoViewGeometry& geometry) {
  return geometry.HasUsableCorrespondences();
}

struct PairJob {
  std::vector<FeatureMatch> raw;
  TwoViewGeometry* geometry = nullptr;
  GuidedDescriptorMatchingStats guided_stats;
  bool usable = false, checked = false, attempted = false, improved = false;
  double matching_seconds = 0.0, geometry_seconds = 0.0, guided_seconds = 0.0;
};

}  // namespace

SiftMatchingPipelineSummary RunSiftMatchingPipeline(
    const std::vector<ImageId>& image_ids,
    const std::vector<ImagePair>& pairs,
    const std::map<ImageId, Camera>& cameras_by_image,
    const PreparedFeatureMatcherCache& prepared,
    const SiftMatchingPipelineOptions& options,
    MatchGraph* graph,
    std::vector<std::vector<FeatureMatch>>* initial_matches) {
  if (graph == nullptr) throw std::invalid_argument("Pipeline requires a graph");
  const auto started = Clock::now();
  std::map<ImageId, std::size_t> image_order;
  for (std::size_t i = 0; i < image_ids.size(); ++i) image_order.emplace(image_ids[i], i);
  for (const auto& pair : pairs) {
    for (ImageId id : {pair.image_id1, pair.image_id2}) {
      if (!image_order.count(id) || !cameras_by_image.count(id) ||
          !prepared.count(id) || !prepared.at(id) || FindFeatures(*graph, id) == nullptr) {
        throw std::invalid_argument("Incomplete SIFT pipeline inputs");
      }
    }
  }
  std::vector<PairJob> jobs(pairs.size());
  ProgressCounter progress("matching_pipeline", pairs.size());
  // Construct each geometry directly in the eventual output container. Its
  // address stays stable during insertion and guided search. Keep this map
  // private until every worker joins so exceptions leave graph unchanged.
  std::map<ImagePair, TwoViewGeometry> geometries;
  std::mutex geometries_mutex;
  PairQueue geometry_queue, guided_queue;
  std::mutex exception_mutex;
  std::exception_ptr first_exception;
  std::atomic<bool> failed{false};
  const int worker_count = std::max(1, parallel::getNumThreads());
  std::atomic<int> workers_left{worker_count};
  auto record_exception = [&] {
    std::lock_guard<std::mutex> lock(exception_mutex);
    if (!first_exception) first_exception = std::current_exception();
    failed.store(true);
  };
  const auto& read_only_graph = *graph;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  std::thread guided_worker;
  try {
    for (int i = 0; i < worker_count; ++i) {
      workers.emplace_back([&] {
        try {
          std::size_t index;
          while (geometry_queue.Pop(&index)) {
            if (failed.load()) continue;
            auto& job = jobs[index];
            const auto& pair = pairs[index];
            const auto begin = Clock::now();
            auto geometry = EstimateTwoViewGeometry(
                cameras_by_image.at(pair.image_id1), *FindFeatures(read_only_graph, pair.image_id1),
                cameras_by_image.at(pair.image_id2), *FindFeatures(read_only_graph, pair.image_id2),
                job.raw, options.geometry.geometry_options);
            job.geometry_seconds = Elapsed(begin);
            job.usable = Usable(geometry);
            if (job.usable || options.geometry.keep_degenerate) {
              const std::lock_guard<std::mutex> lock(geometries_mutex);
              const auto [it, inserted] = geometries.emplace(
                  ImagePair::Ordered(pair.image_id1, pair.image_id2), std::move(geometry));
              if (!inserted) throw std::invalid_argument("Duplicate SIFT pipeline pair");
              job.geometry = &it->second;
            }
            const auto distance = pair.image_id2 > pair.image_id1
                ? pair.image_id2 - pair.image_id1 : pair.image_id1 - pair.image_id2;
            job.checked = options.guided_matching &&
                (options.guided.nearby_image_window == 0 || distance <= options.guided.nearby_image_window);
            if (job.checked && job.usable && HasGuidedMatchingModel(*job.geometry)) guided_queue.Push(index);
            else progress.Advance();
          }
        } catch (...) { record_exception(); }
        if (workers_left.fetch_sub(1) == 1) guided_queue.Close();
      });
    }
    guided_worker = std::thread([&] {
      try {
        std::size_t index;
        while (guided_queue.Pop(&index)) {
          if (failed.load()) continue;
          auto& job = jobs[index];
          const auto& pair = pairs[index];
          const auto begin = Clock::now();
          auto matches = MatchPreparedFeaturePairGuided(
              cameras_by_image.at(pair.image_id1), *FindFeatures(read_only_graph, pair.image_id1), *prepared.at(pair.image_id1),
              cameras_by_image.at(pair.image_id2), *FindFeatures(read_only_graph, pair.image_id2), *prepared.at(pair.image_id2), *job.geometry,
              options.guided.guided_matching.max_sampson_error_px,
              options.guided.relaxed_matching, &job.guided_stats);
          job.guided_seconds = Elapsed(begin);
          job.attempted = true;
          progress.Advance();
          const auto old_count = job.geometry->inlier_matches.size();
          if (matches.size() < old_count + options.guided.min_inlier_gain) continue;
          const double old_ratio = job.raw.empty() ? 0.0 : static_cast<double>(old_count) / job.raw.size();
          const double new_ratio = matches.empty() ? 0.0 : 1.0;
          if (new_ratio + 1e-12 < old_ratio) continue;
          job.geometry->inlier_matches = std::move(matches);
          job.improved = true;
        }
      } catch (...) { record_exception(); }
    });
    for (std::size_t index = 0; index < pairs.size() && !failed.load(); ++index) {
      auto matching = options.matching;
      const auto& pair = pairs[index];
      const auto first = image_order.at(pair.image_id1), second = image_order.at(pair.image_id2);
      const auto distance = first > second ? first - second : second - first;
      if (options.nearby_image_window > 0 && distance <= options.nearby_image_window) {
        matching.max_ratio = options.nearby_max_ratio;
        matching.mutual_check = true;
      }
      const auto begin = Clock::now();
      jobs[index].raw = MatchPreparedFeatureSets(
          *prepared.at(pair.image_id1), *prepared.at(pair.image_id2), matching);
      jobs[index].matching_seconds = Elapsed(begin);
      geometry_queue.Push(index);
    }
  } catch (...) {
    record_exception();
    guided_queue.Close();
  }
  geometry_queue.Close();
  for (auto& worker : workers) worker.join();
  if (guided_worker.joinable()) guided_worker.join();
  if (first_exception) std::rethrow_exception(first_exception);

  SiftMatchingPipelineSummary summary;
  graph->matches.clear();
  graph->verified_geometries = std::move(geometries);
  if (initial_matches) initial_matches->resize(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    auto& job = jobs[index];
    const auto& pair = pairs[index];
    summary.num_initial_matches += job.raw.size();
    if (options.retain_matches) {
      SetMatches(graph, pair.image_id1, pair.image_id2,
                 job.improved ? job.geometry->inlier_matches : job.raw);
    }
    if (initial_matches) (*initial_matches)[index] = std::move(job.raw);
    summary.matching_work_seconds += job.matching_seconds;
    summary.geometry_work_seconds += job.geometry_seconds;
    summary.guided_work_seconds += job.guided_seconds;
    summary.guided.num_checked_pairs += job.checked;
    summary.guided.num_rematched_pairs += job.attempted;
    summary.guided.num_improved_pairs += job.improved;
    summary.guided.candidate_generation_work_seconds += job.guided_stats.candidate_generation_seconds;
    summary.guided.descriptor_search_work_seconds += job.guided_stats.descriptor_search_seconds;
  }
  summary.wall_seconds = Elapsed(started);
  return summary;
}

}  // namespace colite
