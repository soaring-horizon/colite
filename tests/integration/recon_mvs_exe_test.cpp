#include "colite/exe/recon_mvs.h"
#include "colite/exe/mvs_stages.h"
#include "colite/mvs/dense_map.h"
#include "colite/mvs/patch_match.h"
#include "colite/mvs/meshing.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace colite::exe {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

void WriteSmallColmapTextModel(const std::filesystem::path& model_path) {
  std::filesystem::create_directories(model_path);
  {
    std::ofstream cameras(model_path / "cameras.txt");
    cameras << "# Camera list\n"
            << "1 PINHOLE 100 80 50 50 50 40\n";
  }
  {
    std::ofstream images(model_path / "images.txt");
    images << "# Image list\n"
           << "1 1 0 0 0 0 0 0 1 image1.jpg\n"
           << "50 40 1 52.5 40 2\n"
           << "2 1 0 0 0 0 0 0 1 image2.jpg\n"
           << "50 40 1 52.5 40 2\n";
  }
  {
    std::ofstream points(model_path / "points3D.txt");
    points << "# 3D point list\n"
           << "1 0 0 2 255 255 255 0 1 0 2 0\n"
           << "2 0.2 0 4 255 255 255 0 1 1 2 1\n";
  }
}

void WriteSmallImages(const std::filesystem::path& image_path) {
  std::filesystem::create_directories(image_path);
  cv::Mat image1(5, 5, CV_8UC1);
  for (int row = 0; row < image1.rows; ++row) {
    for (int col = 0; col < image1.cols; ++col) {
      image1.at<std::uint8_t>(row, col) =
          static_cast<std::uint8_t>(row * 20 + col * 10);
    }
  }
  const cv::Mat image2 = image1.clone();
  ASSERT_TRUE(cv::imwrite((image_path / "image1.jpg").string(), image1));
  ASSERT_TRUE(cv::imwrite((image_path / "image2.jpg").string(), image2));
}

// These tiny I/O fixtures use co-located cameras and 5px bitmaps. Keep
// their search small and filters explicit; real release defaults are exercised
// by StageContractTest and the full NAVI default-command regression.
std::vector<std::string> SmallStereoFixtureArguments(std::vector<std::string> args,
                                                     bool pipeline) {
  const auto set_default = [&](const std::string& name, const std::string& value) {
    for (const auto& arg : args) if (arg == name || arg.rfind(name + "=", 0) == 0) return;
    args.insert(args.end(), {name, value});
  };
  for (const auto& option : std::vector<std::pair<std::string, std::string>>{
      {"--PatchMatchStereo.num_iterations", "1"}, {"--PatchMatchStereo.window_radius", "1"},
      {"--PatchMatchStereo.num_samples", "1"}, {"--PatchMatchStereo.geom_consistency", "0"},
      {"--PatchMatchStereo.filter", "0"}, {"--Colite.patch_match_backend", "cpu"},
      {"--Colite.patch_match_bilateral_ncc", "0"}, {"--Colite.patch_match_dynamic_source_selection", "0"},
      {"--Colite.patch_match_directional_hmm", "0"}, {"--Colite.patch_match_checkerboard_rounds", "1"},
      {"--Colite.source_selection_mode", "weighted"}, {"--Colite.resolution_level", "0"}}) {
    set_default(option.first, option.second);
  }
  if (pipeline) set_default("--StereoFusion.min_num_pixels", "1");
  return args;
}

int RunFixtureReconMvs(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  return RunReconMvs(SmallStereoFixtureArguments(args, true), out, err);
}

int RunFixturePatchMatchStereo(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  return RunPatchMatchStereo(SmallStereoFixtureArguments(args, false), out, err);
}

TEST(ReconMvsExeTest, PrintsHelp) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunFixtureReconMvs({"recon_mvs", "--help"}, out, err);

  EXPECT_EQ(exit_code, 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("--input_path"), std::string::npos);
  EXPECT_NE(out.str().find("default 20"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_min_num_valid_sources"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_max_num_aggregated_costs"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_geometric_refinement_iterations"),
            std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.geom_consistency_regularizer"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_propagation_schedule"),
            std::string::npos);
  EXPECT_NE(out.str().find("default checkerboard"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_backend"), std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.gpu_index"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_pyramid_levels"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_perturbation_samples"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.patch_match_directional_hmm"),
            std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.num_samples"),
            std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.ncc_sigma"),
            std::string::npos);
  EXPECT_EQ(out.str().find("--patch_match_ncc_sigma"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.refine_normals_from_depth"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.normal_refinement_max_relative_depth_difference"),
            std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.min_triangulation_angle"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.source_triangulation_angle_weight"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.source_scale_consistency_weight"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.source_coverage_weight"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.source_coverage_grid_size"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.write_debug_maps"), std::string::npos);
  EXPECT_NE(out.str().find("--PatchMatchStereo.num_threads"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.filter_dense_cloud"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.mesh_dense_cloud"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.delaunay_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.mesh_texturer"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.fusion_source_pixel_search_radius"),
            std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconMvsExeTest, RejectsInvalidColmapFusionCacheOptions) {
  for (const std::string value : {"0", "-1", "nan", "inf"}) {
    std::ostringstream out, err;
    EXPECT_NE(RunFixtureReconMvs({"recon_mvs", "--input_path", "unused",
                           "--output_path", "unused", "--StereoFusion.cache_size", value}, out, err), 0);
    EXPECT_NE(err.str().find("cache_size"), std::string::npos);
  }
  std::ostringstream out, err;
  EXPECT_NE(RunFixtureReconMvs({"recon_mvs", "--StereoFusion.use_cache", "yes"}, out, err), 0);
  EXPECT_NE(err.str().find("use_cache"), std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidFinalNccFilterOptions) {
  for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
      {"--PatchMatchStereo.filter", "yes"},
      {"--PatchMatchStereo.filter_min_ncc", "1.1"},
      {"--PatchMatchStereo.filter_min_ncc", "nan"},
      {"--PatchMatchStereo.ncc_sigma", "0"},
      {"--PatchMatchStereo.ncc_sigma", "nan"},
      {"--patch_match_ncc_sigma", "0.6"},
      {"--PatchMatchStereo.filter_min_num_consistent", "-1"},
      {"--PatchMatchStereo.filter_min_triangulation_angle", "181"},
      {"--PatchMatchStereo.filter_geom_consistency_max_cost", "-1"}}) {
    std::ostringstream out, err;
    EXPECT_NE(RunFixtureReconMvs({"recon_mvs", "--input_path", "unused", "--output_path", "unused",
                          pair.first, pair.second}, out, err), 0);
    EXPECT_FALSE(err.str().empty());
  }
  std::ostringstream out, err;
  EXPECT_NE(RunFixtureReconMvs({"recon_mvs", "--input_path", "unused", "--output_path", "unused",
      "--PatchMatchStereo.filter", "1", "--Colite.filter_photometric_max_cost", "0.45"}, out, err), 0);
  EXPECT_NE(err.str().find("Do not combine"), std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidPatchMatchBackend) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunFixtureReconMvs(
      {"recon_mvs",
       "--input_path",
       "model",
       "--output_path",
       "mvs",
       "--Colite.patch_match_backend",
       "metal"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("Invalid PatchMatch backend"), std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidPatchMatchPropagationSchedule) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunFixtureReconMvs(
      {"recon_mvs",
       "--input_path",
       "model",
       "--output_path",
       "mvs",
       "--Colite.patch_match_propagation_schedule",
       "diagonal"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("Invalid PatchMatch propagation schedule"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidThreadCount) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunFixtureReconMvs(
      {"recon_mvs",
       "--input_path",
       "model",
       "--output_path",
       "mvs",
       "--PatchMatchStereo.num_threads",
       "0"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("num_threads must be -1 or positive"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidNormalRefinementDepthThreshold) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunFixtureReconMvs(
      {"recon_mvs",
       "--input_path",
       "model",
       "--output_path",
       "mvs",
       "--Colite.normal_refinement_max_relative_depth_difference",
       "-0.1"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find(
                "--Colite.normal_refinement_max_relative_depth_difference must be "
                "non-negative"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RunsSparseWorkspacePassFromColmapTextModel) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_exe_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--output_path",
                      output_path.string()},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Sparse depth samples: 4"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("num_sparse_depth_samples 4"),
            std::string::npos);
  const std::string source_images = ReadTextFile(output_path / "source_images.tsv");
  EXPECT_NE(source_images.find("target_image_id\tsource_image_id"
                               "\tnum_shared_points"
                               "\tmean_triangulation_angle_degrees"
                               "\ttriangulation_angle_percentile_degrees"
                               "\tmean_scale_ratio\tcoverage_score\tscore"),
            std::string::npos);
  EXPECT_NE(source_images.find("1\t2\t2"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "sparse_depth_samples.tsv")
                .find("1\t1\t50\t40\t2"),
            std::string::npos);
}

TEST(ReconMvsExeTest, InitializesPatchMatchMapsWhenImagePathIsProvided) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_init_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--Colite.resolution_level",
                      "0",
                      "--output_path",
                      output_path.string(),
                      "--Colite.initialize_patch_match", "1"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Initialized PatchMatch images: 2"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("initialized_patch_match_images 2"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.init.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.init.bin"));
}

TEST(ReconMvsExeTest, RunsPatchMatchStereoWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_pm_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1", "--Colite.write_debug_maps", "1",
                      "--Colite.resolution_level",
                      "0",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--Colite.patch_match_pyramid_levels",
                      "2",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--Colite.patch_match_perturbation_samples",
                      "2",
                      "--Colite.filter_photometric_max_cost",
                      "0.5",
                      "--Colite.patch_match_min_num_valid_sources",
                      "1",
                      "--Colite.patch_match_max_num_aggregated_costs",
                      "2",
                      "--Colite.patch_match_propagation_schedule",
                      "four_direction",
                      "--PatchMatchStereo.geom_consistency", "1", "--Colite.patch_match_geometric_refinement_iterations",
                      "1",
                      "--PatchMatchStereo.geom_consistency_regularizer",
                      "0.2",
                      "--Colite.patch_match_geometric_max_relative_depth_error",
                      "0.01",
                      "--Colite.refine_normals_from_depth", "1",
                      "--Colite.normal_refinement_max_relative_depth_difference",
                      "1.0"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("PatchMatch images: 2"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_images 2"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("photometric_filtered_pixels "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.geometric.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.geometric.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "cost_maps" /
                                      "image1.jpg.geometric.txt"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "cost_maps" /
                         "image1.jpg.geometric.txt")
                .find("COLITE_DENSE_COST_MAP_TEXT"),
            std::string::npos);
  // Working dimensions follow the camera model, even when the input bitmap
  // has already been resized. The synthetic bitmap is only 5x5, model 100x80.
  const auto depth = mvs::ReadDenseDepthMapText(
      output_path / "stereo" / "depth_maps" / "image1.jpg.geometric.txt");
  EXPECT_EQ(depth.Width(), 100U);
  EXPECT_EQ(depth.Height(), 80U);
  EXPECT_FALSE(std::filesystem::exists(output_path / "stereo/.colite_refinement"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("valid_cost_pixels_before_filter"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("requested_backend"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("actual_backend"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("pyramid_levels"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("mean_cost_before_filter"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("top_down_sweeps"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("checkerboard_passes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("perturbation_mean_abs_depth_update"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("perturbation_geometric_candidate_rejections"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("source_selection_tests"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("iteration_candidate_evaluations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("iteration_source_selection_acceptances"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("normal_refined_pixels"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_wrapper_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_request_validation_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pyramid_resize_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_single_scale_dispatch_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_result_materialization_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_output_to_vector_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_dense_map_fill_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_normal_refinement_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_valid_cost_count_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_host_initial_normal_upload_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_host_output_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_device_count_queries"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_device_count_cache_hits"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_evaluate_cost_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_texture_reuses"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_buffer_reuses"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_iteration_counter_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_output_map_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_host_uploads"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_texture_uploads"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_output_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_host_upload_bytes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_texture_upload_bytes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_host_staging_bytes"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "patch_match_initial.tsv"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_wrapper_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_host_pack_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_host_output_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_result_materialization_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_output_to_vector_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_dense_map_fill_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_normal_refinement_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_valid_cost_count_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_device_count_queries"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_device_count_cache_hits"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_pool_size"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_pool_slot"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("normal_refined_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_wall_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_sum_cuda_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_overlap_saved_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_overlap_ratio "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_deferred_photometric_stats_images "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("cuda_warmup_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_source_setup_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_stereo_run_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_stereo_wrapper_overhead_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_postprocess_sum_time_ms "),
            std::string::npos);
}

TEST(ReconMvsExeTest, RunsPatchMatchStereoWithTwoThreadsWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_pm_threads_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--PatchMatchStereo.num_threads",
                      "2"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("PatchMatch images: 2"), std::string::npos);
  const std::string summary = ReadTextFile(output_path / "summary.txt");
  EXPECT_NE(summary.find("patch_match_initial_wall_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_sum_cuda_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_overlap_saved_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_overlap_ratio "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "patch_match.tsv"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
            .find("image_id\trequested_backend\tactual_backend"
                  "\tpyramid_levels\titerations\tbilateral_ncc"
                  "\tsigma_spatial"
                  "\tsigma_color\tdynamic_source_selection\tncc_sigma"
                  "\tselection_history_schedule"
                  "\tmin_selection_probability"
                  "\tmax_selection_probability\tvalid_cost_pixels"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesGeometricConsistencyGraphWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_geom_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1", "--PatchMatchStereo.write_consistency_graph", "1", "--Colite.write_cost_maps", "1",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--Colite.check_geometric_consistency", "1",
                      "--Colite.filter_geometric_min_num_consistent_sources",
                      "1"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Geometric consistent pixels: "), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("geometric_consistent_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("geometric_filtered_pixels "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "consistency_graphs" /
                                      "image1.jpg.photometric.bin"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("geometric_consistent_pixels"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesBinaryDenseMapsWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_binary_maps_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1", "--PatchMatchStereo.write_consistency_graph", "1", "--Colite.write_cost_maps", "1",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--Colite.patch_match_backend",
                      "cpu",
                      "--PatchMatchStereo.max_image_size",
                      "8",
                      "--PatchMatchStereo.geom_consistency", "1", "--Colite.patch_match_geometric_refinement_iterations",
                      "1",
                      "--PatchMatchStereo.geom_consistency_regularizer",
                      "0.2",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--Colite.check_geometric_consistency", "1",
                      "--Colite.filter_geometric_min_num_consistent_sources",
                      "1",
                      },
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.geometric.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.geometric.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "consistency_graphs" /
                                      "image1.jpg.geometric.bin"));
  EXPECT_FALSE(std::filesystem::exists(output_path / "stereo" /
                                       "consistency_graphs_colmap" /
                                       "image1.jpg.geometric.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "cost_maps" /
                                      "image1.jpg.geometric.bin"));
  EXPECT_FALSE(std::filesystem::exists(output_path / "stereo/.colite_refinement"));
}

TEST(ReconMvsExeTest, PersistsStandardFilterGraphsWithOptionalDebugText) {
  const auto root = std::filesystem::temp_directory_path() / "colite_mvs_final_filter_test";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "model");
  WriteSmallImages(root / "images");
  for (const std::string debug : {"0", "1"}) {
    const auto output = root / debug;
    std::ostringstream out, err;
    ASSERT_EQ(RunFixtureReconMvs({"recon_mvs", "--input_path", (root / "model").string(),
        "--image_path", (root / "images").string(), "--output_path", output.string(),
        "--Colite.run_patch_match", "1", "--Colite.patch_match_backend", "cpu", "--PatchMatchStereo.max_image_size", "8",
        "--PatchMatchStereo.num_iterations", "1", "--PatchMatchStereo.geom_consistency", "1", "--Colite.patch_match_geometric_refinement_iterations", "1",
        "--PatchMatchStereo.geom_consistency_regularizer", "0.3", "--Colite.write_debug_maps", debug,
        "--PatchMatchStereo.write_consistency_graph", "1",
        "--PatchMatchStereo.filter", "1", "--PatchMatchStereo.ncc_sigma", "0.7",
        "--PatchMatchStereo.filter_min_num_consistent", "1"}, out, err), 0) << err.str();
    const auto graph = mvs::ReadConsistencyGraphColmapBinary(output / "stereo/consistency_graphs/image1.jpg.geometric.bin", {1, 2});
    EXPECT_GT(graph.Width(), 0);
    EXPECT_EQ(graph.NumConsistentPixels(1), 0);  // Flat patches cannot satisfy NCC.
    const auto depth = mvs::ReadDenseDepthMapBinary(output / "stereo/depth_maps/image1.jpg.geometric.bin");
    EXPECT_EQ(depth.ValidPixelCount(), 0);
    EXPECT_GT(mvs::ReadDenseDepthMapBinary(output / "stereo/depth_maps/image1.jpg.photometric.bin").ValidPixelCount(), 0);
    const auto text_path = output / "stereo/depth_maps/image1.jpg.geometric.txt";
    EXPECT_EQ(std::filesystem::exists(text_path), debug == "1");
    if (debug == "1") EXPECT_EQ(mvs::ReadDenseDepthMapText(text_path).Data(), depth.Data());
    EXPECT_NE(ReadTextFile(output / "stereo/patch_match.tsv").find("final_filter_applied"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(output / "stereo/.colite_refinement"));
  }
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, MissingGeometrySourceDoesNotMislabelPhotometricMaps) {
  const auto root = std::filesystem::temp_directory_path() / "colite_mvs_missing_dense_source_test";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "model");
  WriteSmallImages(root / "images");
  {
    std::ofstream images(root / "model/images.txt");
    images << "1 1 0 0 0 0 0 0 1 image1.jpg\n50 40 1 52.5 40 2\n"
           << "2 1 0 0 0 0 0 -5 1 image2.jpg\n50 40 1 52.5 40 2\n";
  }
  std::ostringstream out, err;
  ASSERT_NE(RunFixtureReconMvs({"recon_mvs", "--input_path", (root / "model").string(),
      "--image_path", (root / "images").string(), "--output_path", (root / "mvs").string(),
      "--Colite.run_patch_match", "1", "--Colite.write_cost_maps", "1", "--Colite.resolution_level", "0", "--PatchMatchStereo.num_iterations", "1", "--PatchMatchStereo.geom_consistency", "1", "--Colite.patch_match_geometric_refinement_iterations", "1",
      "--PatchMatchStereo.geom_consistency_regularizer", "0.3"}, out, err), 0) << err.str();
  EXPECT_FALSE(std::filesystem::exists(root / "mvs/stereo/depth_maps/image1.jpg.geometric.bin"));
  const auto depth = mvs::ReadDenseDepthMapBinary(root / "mvs/stereo/depth_maps/image1.jpg.photometric.bin");
  const auto costs = mvs::ReadDenseCostMapBinary(root / "mvs/stereo/cost_maps/image1.jpg.photometric.bin");
  EXPECT_EQ(costs.NumPixels(), depth.NumPixels());
  EXPECT_EQ(costs.NumPixels(), 8000U);
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, ByteBoundedFusionCachePreservesSingleThreadOutput) {
  const auto root = std::filesystem::temp_directory_path() / "colite_mvs_byte_cache_test";
  std::filesystem::remove_all(root);
  const auto model = root / "model", images = root / "images", output = root / "mvs";
  WriteSmallColmapTextModel(model);
  WriteSmallImages(images);
  const std::vector<std::string> base{"recon_mvs", "--input_path", model.string(),
      "--image_path", images.string(), "--output_path", output.string(),
      "--StereoFusion.num_threads", "1",
      "--Colite.resolution_level", "0", "--Colite.fuse_stereo", "1", "--input_type", "photometric"};
  auto initial = base;
  initial.insert(initial.end(), {"--Colite.run_patch_match", "1", "--PatchMatchStereo.num_iterations", "1"});
  std::ostringstream out, err;
  ASSERT_EQ(RunFixtureReconMvs(initial, out, err), 0) << err.str();
  const auto cloud = ReadTextFile(output / "fused.ply");
  const auto source_diagnostics = ReadTextFile(output / "source_images.tsv");
  auto cached = base;
  cached.insert(cached.end(), {"--StereoFusion.use_cache", "true",
      "--StereoFusion.cache_size", "0.0000001", "--Colite.source_selection_mode", "colmap",
      "--PatchMatchStereo.min_triangulation_angle", "90", "--Colite.source_max_triangulation_angle_degrees", "90"});
  ASSERT_EQ(RunFixtureReconMvs(cached, out, err), 0) << err.str();
  EXPECT_TRUE(ReadTextFile(output / "fused.ply") == cloud);
  EXPECT_TRUE(ReadTextFile(output / "source_images.tsv") == source_diagnostics);
  EXPECT_NE(ReadTextFile(output / "dense/fusion_timing.txt").find("dense_view_loads "),
            std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, WritesStereoFusionPlyWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_fusion_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--Colite.fuse_stereo", "1",
                      "--StereoFusion.min_num_pixels",
                      "1",
                      "--StereoFusion.max_normal_error",
                      "10",
                      "--Colite.fusion_max_view_angle_degrees",
                      "180",
                      "--Colite.fusion_diagnostics", "1",
                      "--Colite.filter_geometric_min_num_consistent_sources",
                      "1"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Fused points: "), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt").find("fused_points "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt").find("fusion_tracks "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fused_observations "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_visited_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_rejected_normal_observations "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_rejected_view_observations "),
            std::string::npos);
  const std::filesystem::path ply_path =
      output_path / "fused.ply";
  ASSERT_TRUE(std::filesystem::exists(ply_path));
  EXPECT_TRUE(std::filesystem::exists(output_path / "fused.ply.vis"));
  EXPECT_FALSE(std::filesystem::exists(output_path / "dense/fused.ply"));
  const std::string ply = ReadTextFile(ply_path);
  EXPECT_NE(ply.find("element vertex "), std::string::npos);
  EXPECT_NE(ply.find("property float nx"), std::string::npos);
  EXPECT_NE(ply.find("property uchar red"), std::string::npos);
  const std::filesystem::path diagnostics_path =
      output_path / "dense" / "fusion_tracks.tsv";
  ASSERT_TRUE(std::filesystem::exists(diagnostics_path));
  const std::string diagnostics = ReadTextFile(diagnostics_path);
  EXPECT_NE(diagnostics.find("point_idx\tseed_image_id\tseed_row\tseed_col"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("max_normal_angle_degrees"), std::string::npos);
  EXPECT_NE(diagnostics.find("max_view_angle_degrees"), std::string::npos);

  const std::filesystem::path pair_diagnostics_path =
      output_path / "dense" / "fusion_image_pairs.tsv";
  ASSERT_TRUE(std::filesystem::exists(pair_diagnostics_path));
  const std::string pair_diagnostics = ReadTextFile(pair_diagnostics_path);
  EXPECT_NE(pair_diagnostics.find("reference_image_id\tsource_image_id"
                                  "\tnum_candidates\tnum_accepted"),
            std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_normal"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_view"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_projection"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_depth"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_duplicate"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_visited"), std::string::npos);

  const std::filesystem::path track_lengths_path =
      output_path / "dense" / "fusion_track_lengths.tsv";
  ASSERT_TRUE(std::filesystem::exists(track_lengths_path));
  const std::string track_lengths = ReadTextFile(track_lengths_path);
  EXPECT_NE(track_lengths.find("num_observations\tnum_tracks"),
            std::string::npos);
  const std::filesystem::path fusion_summary_path =
      output_path / "dense" / "fusion_summary.txt";
  ASSERT_TRUE(std::filesystem::exists(fusion_summary_path));
  const std::string fusion_summary = ReadTextFile(fusion_summary_path);
  EXPECT_NE(fusion_summary.find("num_single_observation_tracks "),
            std::string::npos);
  EXPECT_NE(fusion_summary.find("multi_observation_track_ratio "),
            std::string::npos);

  const std::filesystem::path error_distribution_path =
      output_path / "dense" / "fusion_error_distribution.tsv";
  ASSERT_TRUE(std::filesystem::exists(error_distribution_path));
  const std::string error_distribution =
      ReadTextFile(error_distribution_path);
  EXPECT_NE(error_distribution.find("metric\trecord\tlabel\tlower\tupper"
                                    "\tcount\tvalue"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("relative_depth_error\thistogram"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("reprojection_error_pixels\tpercentile"
                                    "\tp95"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("normal_angle_degrees\tpercentile\tp95"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("view_angle_degrees\tpercentile\tp95"),
            std::string::npos);
}

TEST(ReconMvsExeTest, StandaloneFusionReusesPersistedPatchMatchMaps) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_standalone_fusion_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream patch_out;
  std::ostringstream err;
  ASSERT_EQ(RunFixturePatchMatchStereo(
                {"patch_match_stereo", "--input_path", model_path.string(),
                 "--image_path", image_path.string(), "--output_path",
                 output_path.string(), "--PatchMatchStereo.num_iterations", "1",
                 "--PatchMatchStereo.window_radius", "1"},
                patch_out, err),
            0)
      << err.str();
  const auto depth_path = output_path / "stereo" / "depth_maps" /
                          "image1.jpg.photometric.bin";
  ASSERT_TRUE(std::filesystem::exists(depth_path));
  const auto depth_time = std::filesystem::last_write_time(depth_path);
  std::ofstream(output_path / "stereo/fusion.cfg") << "image1.jpg\nimage2.jpg\n";

  std::ostringstream fusion_out;
  err.str("");
  err.clear();
  EXPECT_EQ(RunStereoFusion(
                {"stereo_fusion", "--input_path", model_path.string(),
                 "--image_path", image_path.string(), "--workspace_path", output_path.string(),
                 "--output_path", (output_path / "dense/fused.ply").string(),
                 "--input_type", "photometric", "--StereoFusion.min_num_pixels", "1"},
                fusion_out, err),
            0)
      << err.str();
  EXPECT_EQ(std::filesystem::last_write_time(depth_path), depth_time);
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" / "fused.ply"));
  const std::filesystem::path timing_path =
      output_path / "dense" / "fusion_timing.txt";
  ASSERT_TRUE(std::filesystem::exists(timing_path));
  const std::string timing = ReadTextFile(timing_path);
  EXPECT_NE(timing.find("consistency_graph_loading_ms "), std::string::npos);
  EXPECT_NE(timing.find("fusion_core_ms "), std::string::npos);
  EXPECT_NE(timing.find("total_ms "), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(
      output_path / ".colite" / "stages" / "patch_match_stereo.json"));
  EXPECT_TRUE(std::filesystem::exists(
      output_path / ".colite" / "stages" / "stereo_fusion.json"));

  std::ostringstream resume_out;
  err.str("");
  err.clear();
  EXPECT_EQ(RunStereoFusion(
                {"stereo_fusion", "--input_path", model_path.string(),
                 "--image_path", image_path.string(), "--workspace_path", output_path.string(),
                 "--output_path", (output_path / "dense/fused.ply").string(),
                 "--input_type", "photometric", "--StereoFusion.min_num_pixels", "1",
                 "--resume"},
                resume_out, err),
            0)
      << err.str();
  EXPECT_NE(resume_out.str().find("SKIP stereo_fusion"), std::string::npos);
}

TEST(ReconMvsExeTest, StandardMapsAreDefaultAndResumeRepairsMissingMembers) {
  const auto root = std::filesystem::temp_directory_path() / "colite_standard_map_contract";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  std::filesystem::create_directories(root / "stereo");
  std::ofstream(root / "stereo/patch-match.cfg") << "image1.jpg\nimage2.jpg\nimage2.jpg\nimage1.jpg\n";
  std::vector<std::string> command{"patch_match_stereo", "--workspace_path", root.string(),
      "--Colite.patch_match_backend", "cpu", "--PatchMatchStereo.num_threads", "1",
      "--PatchMatchStereo.max_image_size", "8", "--PatchMatchStereo.num_iterations", "1",
      "--PatchMatchStereo.window_radius", "1", "--Colite.resolution_level", "0"};
  std::ostringstream out, err;
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  const auto depth = root / "stereo/depth_maps/image1.jpg.photometric.bin";
  const auto normal = root / "stereo/normal_maps/image1.jpg.photometric.bin";
  EXPECT_EQ(ReadTextFile(depth).substr(0, 6), "8&6&1&");
  EXPECT_EQ(ReadTextFile(normal).substr(0, 6), "8&6&3&");
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.bin"));
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.photometric.txt"));
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/cost_maps/image1.jpg.photometric.bin"));
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/consistency_graphs/image1.jpg.photometric.bin"));
  command.push_back("--resume");
  out.str("");
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("SKIP patch_match_stereo"), std::string::npos);
  std::filesystem::remove(normal);
  out.str("");
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  EXPECT_EQ(out.str().find("SKIP "), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(normal));
  auto obsolete = command;
  obsolete.insert(obsolete.end(), {"--Colite.dense_map_format", "binary"});
  EXPECT_NE(RunFixturePatchMatchStereo(obsolete, out, err), 0);
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, PipelineInvalidatesOwnedPhotometricMapsWhenInputsOrParametersChange) {
  const auto root = std::filesystem::temp_directory_path() / "colite_pipeline_photo_ownership";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  std::filesystem::create_directories(root / "stereo");
  std::ofstream(root / "stereo/patch-match.cfg")
      << "image1.jpg\nimage2.jpg\nimage2.jpg\nimage1.jpg\n";
  std::vector<std::string> command{
      "recon_mvs", "--input_path", (root / "sparse").string(),
      "--image_path", (root / "images").string(), "--output_path", root.string(),
      "--Colite.run_patch_match", "1", "--PatchMatchStereo.geom_consistency", "1",
      "--PatchMatchStereo.max_image_size", "8", "--PatchMatchStereo.num_threads", "1"};
  std::ostringstream out, err;
  const auto reused = [&] {
    return ReadTextFile(root / "stereo/patch_match_initial.tsv").find("persisted") != std::string::npos;
  };
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_FALSE(reused());
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_TRUE(reused());
  command.insert(command.end(), {"--default_random_seed", "43"});
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_FALSE(reused());  // New random search must not inherit the old seed's photo maps.
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_TRUE(reused());
  cv::Mat changed(5, 5, CV_8UC1, cv::Scalar(235));
  ASSERT_TRUE(cv::imwrite((root / "images/image1.jpg").string(), changed));
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_FALSE(reused());
  std::ofstream(root / "sparse/cameras.txt") << "1 PINHOLE 100 80 55 55 50 40\n";
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_FALSE(reused());
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_TRUE(reused());
  // A fusion-only option must not invalidate an unchanged photometric search.
  command.insert(command.end(), {"--StereoFusion.max_depth_error", "0.02"});
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_TRUE(reused());
  auto standalone = command;
  standalone.front() = "patch_match_stereo";
  for (const auto& name : {"--Colite.run_patch_match", "--StereoFusion.max_depth_error"}) {
    const auto it = std::find(standalone.begin(), standalone.end(), name);
    ASSERT_NE(it, standalone.end());
    standalone.erase(it, it + 2);
  }
  ASSERT_EQ(RunFixturePatchMatchStereo(standalone, out, err), 0) << err.str();
  EXPECT_TRUE(reused());  // Changing the entry point preserves the resolved key.
  standalone.push_back("--resume");
  out.str("");
  ASSERT_EQ(RunFixturePatchMatchStereo(standalone, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("SKIP "), std::string::npos);
  const auto ownership = root / ".colite/photometric_cache.txt";
  auto interrupted = ReadTextFile(ownership);
  const auto complete = interrupted.find("complete");
  ASSERT_NE(complete, std::string::npos);
  interrupted.replace(complete, std::string("complete").size(), "running");
  std::ofstream(ownership) << interrupted;
  out.str("");
  ASSERT_EQ(RunFixturePatchMatchStereo(standalone, out, err), 0) << err.str();
  EXPECT_EQ(out.str().find("SKIP "), std::string::npos);
  EXPECT_FALSE(reused());  // An interrupted replacement must not bless old maps.
  std::filesystem::remove(ownership);
  ASSERT_EQ(RunFixtureReconMvs(command, out, err), 0) << err.str();
  EXPECT_FALSE(reused());  // Legacy Colite output has no trustworthy identity.
}

TEST(ReconMvsExeTest, GeometricSubsetReusesStandardMapsAndFullModelGraphIndices) {
  const auto root = std::filesystem::temp_directory_path() / "colite_geometric_import_contract";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  std::ofstream(root / "sparse/images.txt")
      << "2 1 0 0 0 0 0 0 1 image2.jpg\n50 40 1 52.5 40 2\n"
      << "1 1 0 0 0 0 0 0 1 image1.jpg\n50 40 1 52.5 40 2\n";
  mvs::DenseDepthMap depth(8, 6); depth.Fill(3);
  mvs::NormalMap normal(8, 6); normal.Fill(Eigen::Vector3f(0, 0, -1));
  for (const auto* name : {"image1.jpg", "image2.jpg"}) {
    const auto file = std::string(name) + ".photometric.bin";
    mvs::WriteDenseDepthMapColmapBinary(root / "stereo/depth_maps" / file, depth);
    mvs::WriteNormalMapColmapBinary(root / "stereo/normal_maps" / file, normal);
  }
  std::ofstream(root / "stereo/patch-match.cfg") << "image2.jpg\nimage1.jpg\n";
  const auto input = root / "stereo/depth_maps/image2.jpg.photometric.bin";
  const auto before = ReadTextFile(input);
  const auto modified = std::filesystem::last_write_time(input);
  std::vector<std::string> command{"patch_match_stereo", "--workspace_path", root.string(),
      "--Colite.patch_match_backend", "cpu", "--PatchMatchStereo.num_threads", "1",
      "--PatchMatchStereo.max_image_size", "8", "--PatchMatchStereo.num_iterations", "1",
      "--PatchMatchStereo.window_radius", "1", "--Colite.resolution_level", "0",
      "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.geom_consistency_regularizer", "0",
      "--PatchMatchStereo.write_consistency_graph", "1", "--Colite.check_geometric_consistency", "1"};
  std::ostringstream out, err;
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  EXPECT_EQ(ReadTextFile(input), before);
  EXPECT_EQ(std::filesystem::last_write_time(input), modified);
  EXPECT_NE(ReadTextFile(root / "stereo/patch_match_initial.tsv").find("persisted"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.geometric.bin"));
  const auto geometry = root / "stereo/depth_maps/image2.jpg.geometric.bin";
  EXPECT_TRUE(std::filesystem::exists(geometry));
  const auto graph = mvs::ReadConsistencyGraphColmapBinary(
      root / "stereo/consistency_graphs/image2.jpg.geometric.bin", {2, 1});
  ASSERT_GT(graph.NumConsistentPixels(1), 0);
  for (std::size_t y = 0; y < graph.Height(); ++y) for (std::size_t x = 0; x < graph.Width(); ++x) {
    for (const auto id : graph.SourceImageIds(y, x)) EXPECT_EQ(id, 1);
  }
  auto filtered = command;
  filtered.insert(filtered.end(), {"--PatchMatchStereo.filter", "1", "--PatchMatchStereo.filter_min_ncc", "-1",
      "--PatchMatchStereo.filter_min_triangulation_angle", "0", "--Colite.source_geometric_priors", "0"});
  ASSERT_EQ(RunFixturePatchMatchStereo(filtered, out, err), 0) << err.str();
  // The configured default is two sources; COLMAP clips it to this one available source.
  EXPECT_GT(mvs::ReadDenseDepthMapBinary(geometry).ValidPixelCount(), 0);
  auto changed = command;
  changed.insert(changed.end(), {"--default_random_seed", "43", "--resume"});
  ASSERT_EQ(RunFixturePatchMatchStereo(changed, out, err), 0) << err.str();
  EXPECT_EQ(ReadTextFile(root / "stereo/patch_match_initial.tsv").find("persisted"), std::string::npos);
  EXPECT_NE(ReadTextFile(input), before);  // Changed owned configuration cannot borrow an obsolete photo map.
  auto forced = command;
  forced.push_back("--force");
  ASSERT_EQ(RunFixturePatchMatchStereo(forced, out, err), 0) << err.str();
  EXPECT_EQ(ReadTextFile(root / "stereo/patch_match_initial.tsv").find("persisted"), std::string::npos);
  EXPECT_NE(ReadTextFile(input), before);
  const auto reference_normal = root / "stereo/normal_maps/image2.jpg.photometric.bin";
  std::filesystem::resize_file(reference_normal, std::filesystem::file_size(reference_normal) - 1);
  auto resume = command;
  resume.push_back("--resume");
  out.str("");
  ASSERT_EQ(RunFixturePatchMatchStereo(resume, out, err), 0) << err.str();
  EXPECT_EQ(out.str().find("SKIP "), std::string::npos);
  EXPECT_EQ(ReadTextFile(root / "stereo/patch_match_initial.tsv").find("persisted"), std::string::npos);
  EXPECT_EQ(mvs::ReadNormalMapBinary(reference_normal).Width(), 8);
  std::filesystem::remove(geometry);
  std::filesystem::remove(root / "stereo/normal_maps/image1.jpg.photometric.bin");
  EXPECT_NE(RunFixturePatchMatchStereo(command, out, err), 0);
  EXPECT_FALSE(std::filesystem::exists(geometry));
  EXPECT_NE(err.str().find("Missing photometric"), std::string::npos);
  command.insert(command.end(), {"--PatchMatchStereo.allow_missing_files", "1"});
  EXPECT_NE(RunFixturePatchMatchStereo(command, out, err), 0);  // No usable geometric source remains.
  EXPECT_FALSE(std::filesystem::exists(geometry));
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, FusionAlignsImageCameraAndDifferentMapSizesWithoutSparsePoints) {
  const auto root = std::filesystem::temp_directory_path() / "colite_fusion_scaled_map_contract";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "sparse");
  std::filesystem::create_directories(root / "images");
  std::ofstream(root / "sparse/cameras.txt") << "1 PINHOLE 16 16 16 16 8 8\n";
  std::ofstream(root / "sparse/images.txt") << "1 1 0 0 0 0 0 0 1 a.png\n\n";
  std::ofstream(root / "sparse/points3D.txt");
  cv::Mat image(16, 16, CV_8UC3);
  for (int row = 0; row < 16; ++row) for (int col = 0; col < 16; ++col) {
    image.at<cv::Vec3b>(row, col) = cv::Vec3b(60, row * 15, col * 15);
  }
  ASSERT_TRUE(cv::imwrite((root / "images/a.png").string(), image));
  std::filesystem::create_directories(root / "stereo");
  std::ofstream(root / "stereo/fusion.cfg") << "a.png\n";
  for (const int map_width : {8, 32}) {
  mvs::DenseDepthMap depth(map_width, map_width); depth.Fill(3);
  mvs::NormalMap normal(map_width, map_width); normal.Fill(Eigen::Vector3f(0, 0, -1));
  mvs::WriteDenseDepthMapColmapBinary(root / "stereo/depth_maps/a.png.photometric.bin", depth);
  mvs::WriteNormalMapColmapBinary(root / "stereo/normal_maps/a.png.photometric.bin", normal);
  for (const int size : {-1, 4}) {
    const auto output = root / (std::to_string(map_width) + "_" + std::to_string(size) + ".ply");
    std::ostringstream out, err;
    ASSERT_EQ(RunStereoFusion({"stereo_fusion", "--workspace_path", root.string(),
        "--output_path", output.string(), "--input_type", "photometric",
        "--StereoFusion.min_num_pixels", "1", "--StereoFusion.num_threads", "1",
        "--StereoFusion.max_image_size", std::to_string(size),
        "--Colite.resolution_level", "0"}, out, err), 0) << err.str();
    const auto cloud = mvs::ReadDenseTriangleMeshPly(output);
    const int width = size == -1 ? std::min(map_width, 16) : 4;
    ASSERT_EQ(cloud.vertices.size(), width * width);
    for (const auto& point : cloud.vertices) {
      EXPECT_NEAR(point.xyz.z(), 3, 1e-6);
      EXPECT_NEAR(point.normal.z(), -1, 1e-6);
      const int col = static_cast<int>(std::lround(point.xyz.x() / 3 * width + width / 2));
      const int row = static_cast<int>(std::lround(point.xyz.y() / 3 * width + width / 2));
      EXPECT_GE(col, 0); EXPECT_LT(col, width);
      EXPECT_GE(row, 0); EXPECT_LT(row, width);
      if (size == -1) {
        EXPECT_EQ(point.color.x(), std::min(15, static_cast<int>(std::lround(col * 16.0 / width))) * 15);
        EXPECT_EQ(point.color.y(), std::min(15, static_cast<int>(std::lround(row * 16.0 / width))) * 15);
      }
      EXPECT_EQ(point.color.z(), 60);
    }
    const auto visibility = mvs::ReadPointsVisibilityColmapBinary(output.string() + ".vis");
    ASSERT_EQ(visibility.size(), cloud.vertices.size());
    for (const auto& ids : visibility) EXPECT_EQ(ids, (std::vector<std::uint32_t>{0}));
  }
  }
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, PatchMatchReadsReferenceSubsetAndExplicitSourcesWithoutSparsePoints) {
  const auto root = std::filesystem::temp_directory_path() / "colite_patch_match_config_contract";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  std::ofstream(root / "sparse/points3D.txt");
  std::filesystem::create_directories(root / "stereo");
  std::ofstream(root / "stereo/patch-match.cfg") << "# reference subset\n\nimage2.jpg\n image1.jpg \n";
  std::vector<std::string> command{"patch_match_stereo", "--workspace_path", root.string(),
      "--PatchMatchStereo.depth_min", "2", "--PatchMatchStereo.depth_max", "4",
      "--PatchMatchStereo.num_iterations", "1", "--PatchMatchStereo.window_radius", "1",
      "--PatchMatchStereo.num_threads", "1", "--Colite.patch_match_backend", "cpu",
      "--Colite.resolution_level", "0"};
  std::ostringstream out, err;
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("PatchMatch images: 1"), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(root / "stereo/depth_maps/image2.jpg.photometric.bin"));
  EXPECT_FALSE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.photometric.bin"));
  // Malformed configurations fail before writing new reference maps.
  for (const auto& config : {"image1.jpg\n", "image1.jpg\nmissing.jpg\n",
       "image1.jpg\nimage1.jpg\n", "image1.jpg\nimage2.jpg,image2.jpg\n",
       "image1.jpg\nimage2.jpg,\n", "image1.jpg\n__auto__,2junk\n",
       "image1.jpg\n__all__\nimage1.jpg\n__all__\n"}) {
    std::ofstream(root / "stereo/patch-match.cfg") << config;
    EXPECT_NE(RunFixturePatchMatchStereo(command, out, err), 0) << config;
    EXPECT_FALSE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.photometric.bin"));
  }
  std::ofstream(root / "chosen.cfg") << "image1.jpg\n__all__\n";
  command.insert(command.end(), {"--config_path", (root / "chosen.cfg").string(), "--resume"});
  ASSERT_EQ(RunFixturePatchMatchStereo(command, out, err), 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(root / "stereo/depth_maps/image1.jpg.photometric.bin"));
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, FusionConfigSubsetKeepsFullModelVisibilityIndices) {
  const auto root = std::filesystem::temp_directory_path() / "colite_fusion_config_contract";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  mvs::DenseDepthMap depth(5, 5);
  mvs::NormalMap normal(5, 5);
  for (std::size_t y = 0; y < 5; ++y) for (std::size_t x = 0; x < 5; ++x) {
    depth.Set(y, x, 3);
    normal.SetNormalized(x, y, Eigen::Vector3f(0, 0, -1));
  }
  mvs::WriteDenseDepthMapColmapBinary(root / "stereo/depth_maps/image2.jpg.photometric.bin", depth);
  mvs::WriteNormalMapColmapBinary(root / "stereo/normal_maps/image2.jpg.photometric.bin", normal);
  std::ofstream(root / "stereo/fusion.cfg") << "image2.jpg\n";
  std::vector<std::string> command{"stereo_fusion", "--workspace_path", root.string(),
      "--output_path", (root / "fused.ply").string(), "--input_type", "photometric",
      "--StereoFusion.min_num_pixels", "1", "--StereoFusion.num_threads", "1",
      "--Colite.resolution_level", "0"};
  std::ostringstream out, err;
  ASSERT_EQ(RunStereoFusion(command, out, err), 0) << err.str();
  const auto visibility = mvs::ReadPointsVisibilityColmapBinary(root / "fused.ply.vis");
  ASSERT_EQ(visibility.size(), 25);
  for (const auto& indices : visibility) EXPECT_EQ(indices, std::vector<std::uint32_t>{1});
  // COLMAP indices follow file records, not numeric image-ID sorting.
  std::ofstream(root / "sparse/images.txt")
      << "2 1 0 0 0 0 0 0 1 image2.jpg\n50 40 1 52.5 40 2\n"
      << "1 1 0 0 0 0 0 0 1 image1.jpg\n50 40 1 52.5 40 2\n";
  ASSERT_EQ(RunStereoFusion(command, out, err), 0) << err.str();
  for (const auto& indices : mvs::ReadPointsVisibilityColmapBinary(root / "fused.ply.vis")) {
    EXPECT_EQ(indices, std::vector<std::uint32_t>{0});
  }
  // cfg is an input of resume; changing it cannot reuse a previous success.
  std::ofstream(root / "stereo/fusion.cfg") << "missing.jpg\n";
  command.push_back("--resume");
  out.str("");
  EXPECT_NE(RunStereoFusion(command, out, err), 0);
  EXPECT_EQ(out.str().find("SKIP "), std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, WritesUndistortedWorkspaceWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_undistorter_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = RunFixtureReconMvs(
      {"recon_mvs", "--input_path", model_path.string(), "--image_path",
       image_path.string(), "--output_path", output_path.string(),
       "--Colite.resolution_level", "0",
       "--Colite.image_undistorter", "1", "--max_image_size", "64",
       "--Colite.initialize_patch_match", "1"},
      out, err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "images" /
                                      "image1.jpg"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "images" /
                                      "image2.jpg"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "sparse" /
                                      "cameras.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "sparse" /
                                      "images.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "sparse" /
                                      "points3D.txt"));
  EXPECT_NE(ReadTextFile(output_path / "sparse" /
                         "cameras.txt")
                .find("PINHOLE 64 51"),
            std::string::npos);
  EXPECT_NE(out.str().find("Undistorted images: 2"), std::string::npos);
  EXPECT_EQ(ReadTextFile(output_path / "stereo/patch-match.cfg"),
            "image1.jpg\n__auto__, 20\nimage2.jpg\n__auto__, 20\n");
  EXPECT_EQ(ReadTextFile(output_path / "stereo/fusion.cfg"), "image1.jpg\nimage2.jpg\n");
  EXPECT_NE(out.str().find("Initialized PatchMatch images: 2"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.init.bin"));
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, ExecutesRequestedSurfaceStepsAfterPersistedFusion) {
  if (!mvs::IsPoissonMeshingAvailable() || !mvs::IsDelaunayMeshingAvailable()) {
    GTEST_SKIP() << "This integration exercises both optional mesh backends";
  }
  const auto root = std::filesystem::temp_directory_path() /
                    "colite_recon_mvs_surface_pipeline_test";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(root / "sparse");
  WriteSmallImages(root / "images");
  std::ostringstream out, err;
  const int result = RunFixtureReconMvs({"recon_mvs", "--input_path", (root / "sparse").string(),
      "--image_path", (root / "images").string(), "--output_path", (root / "mvs").string(),
      "--Colite.run_patch_match", "1", "--Colite.fuse_stereo", "1",
      "--StereoFusion.min_num_pixels", "1", "--StereoFusion.num_threads", "1",
      "--Colite.poisson_mesher", "1", "--PoissonMeshing.depth", "4", "--PoissonMeshing.trim", "0",
      "--PoissonMeshing.num_threads", "1", "--DelaunayMeshing.num_threads", "1",
      "--Colite.delaunay_mesher", "1", "--Colite.mesh_dense_cloud", "1",
      "--Colite.dense_cloud_min_track_observations", "1",
      "--Colite.mesh_max_edge_length", "10", "--Colite.mesh_max_normal_angle_degrees", "180",
      "--Colite.mesh_simplifier", "1", "--Colite.mesh_texturer", "1",
      "--MeshTextureMapping.num_threads", "1", "--MeshTextureMapping.texture_scale_factor", "0.1",
      "--Colite.texture_atlas_size", "64", "--Colite.texture_max_atlas_size", "512",
      "--MeshTextureMapping.atlas_patch_padding", "1"}, out, err);
  ASSERT_EQ(result, 0) << err.str();
  const auto dense = root / "mvs/dense";
  for (const auto* name : {"meshed-poisson.ply", "meshed-delaunay.ply", "mesh.ply",
                          "meshed-poisson-simplified.ply", "meshed-delaunay-simplified.ply",
                          "mesh-simplified.ply", "textured-poisson/mesh.ply",
                          "textured-delaunay/mesh.ply", "textured-mesh/mesh.ply"}) {
    EXPECT_TRUE(std::filesystem::exists(dense / name)) << name;
  }
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, WritesCleanDenseCloudAndMeshWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_clean_mesh_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunFixtureReconMvs({"recon_mvs",
                      "--input_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--Colite.run_patch_match", "1",
                      "--PatchMatchStereo.num_iterations",
                      "1",
                      "--PatchMatchStereo.window_radius",
                      "1",
                      "--Colite.fuse_stereo", "1",
                      "--StereoFusion.min_num_pixels",
                      "1",
                      "--Colite.filter_geometric_min_num_consistent_sources",
                      "1",
                      "--Colite.filter_dense_cloud", "1",
                      "--Colite.dense_cloud_min_track_observations",
                      "1",
                      "--Colite.dense_cloud_voxel_size",
                      "0",
                      "--Colite.mesh_dense_cloud", "1",
                      "--Colite.mesh_max_edge_length",
                      "10",
                      "--Colite.mesh_max_normal_angle_degrees",
                      "180"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "fused_clean.ply"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "fused_clean_stats.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" / "mesh.ply"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "mesh_stats.txt"));
  EXPECT_NE(ReadTextFile(output_path / "dense" / "fused_clean_stats.txt")
                .find("num_filtered_points "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "dense" / "mesh.ply")
                .find("element face "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "dense" / "mesh_stats.txt")
                .find("num_triangles "),
            std::string::npos);
}

}  // namespace
}  // namespace colite::exe
