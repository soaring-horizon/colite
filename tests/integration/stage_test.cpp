#include "colite/exe/stage.h"
#include "colite/exe/mvs_stages.h"
#include "colite/exe/recon_mvs.h"
#include "colite/exe/recon_auto.h"
#include "colite/exe/mvs_tools.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace colite::exe {
namespace {

class StageContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root = std::filesystem::temp_directory_path() /
           ("colite_stage_contract_" +
            std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
  }
  void TearDown() override { std::filesystem::remove_all(root); }

  void Write(const std::filesystem::path& path, const std::string& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << value;
  }
  std::filesystem::path root;
};

TEST_F(StageContractTest, TexturerAcceptsColmapWorkspaceAndOutputType) {
  std::ostringstream out, err;
  EXPECT_EQ(RunMeshTexturer({"mesh_texturer", "--workspace_path", root.string(),
      "--input_path", (root / "mesh.ply").string(), "--output_path", (root / "texture").string(),
      "--output_type", "TXT", "--dry_run"}, out, err), 0) << err.str();
  for (const auto& extra : {std::vector<std::string>{"--output_type", "OBJ"},
        {"--model_path", "sparse"}, {"--image_path", "images"}, {"--MeshTextureMapping.atlas_size", "64"}}) {
    std::vector<std::string> args{"mesh_texturer", "--workspace_path", root.string(),
        "--input_path", "mesh.ply", "--output_path", (root / "texture").string(), "--dry_run"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::ostringstream rejected_out, rejected_err;
    EXPECT_NE(RunMeshTexturer(args, rejected_out, rejected_err), 0);
    EXPECT_TRUE(rejected_out.str().empty());
  }
  std::ostringstream auto_out, auto_err;
  EXPECT_NE(RunReconAuto({"recon_auto", "--image_path", "images", "--output_path", root.string(),
      "--stop_after", "sfm", "--Colite.texture_output_type", "OBJ", "--dry_run"}, auto_out, auto_err), 0);
  EXPECT_TRUE(auto_out.str().empty());
  auto_out.str(""); auto_err.str("");
  EXPECT_EQ(RunReconAuto({"recon_auto", "--image_path", "images", "--output_path", root.string(),
      "--Colite.texture_output_type", "txt", "--Colite.texture_atlas_size", "64", "--dry_run"}, auto_out, auto_err), 0) << auto_err.str();
  EXPECT_NE(auto_out.str().find("PLAN mesh_texturer"), std::string::npos);
}

TEST_F(StageContractTest, SimplifierAcceptsColmapCanonicalOptions) {
  std::ostringstream out, err;
  EXPECT_EQ(RunMeshSimplifier({"mesh_simplifier", "--input_path", (root / "input.ply").string(),
      "--output_path", (root / "output.ply").string(), "--MeshSimplification.target_face_ratio", "0.1",
      "--MeshSimplification.boundary_weight", "1000", "--MeshSimplification.interpolate_colors", "false",
      "--MeshSimplification.num_threads", "2", "--dry_run"}, out, err), 0) << err.str();
  const auto options = ParseReconMvsArguments({"recon_mvs", "--input_path", "sparse",
      "--image_path", "images", "--output_path", root.string()});
  const auto defaults = options.SimplificationOptions();
  EXPECT_DOUBLE_EQ(defaults.target_face_ratio, .1);
  EXPECT_DOUBLE_EQ(defaults.boundary_weight, 1000);
  EXPECT_TRUE(defaults.interpolate_colors);
  EXPECT_EQ(defaults.num_threads, -1);
  for (const auto* name : {"--MeshSimplification.target_ratio", "--MeshSimplification.target_num_triangles",
                          "--MeshSimplification.preserve_boundary", "--MeshSimplification.max_normal_change"}) {
    std::ostringstream rejected_out, rejected_err;
    EXPECT_NE(RunMeshSimplifier({"mesh_simplifier", "--input_path", "input.ply", "--output_path", "output.ply",
        name, "1", "--dry_run"}, rejected_out, rejected_err), 0);
    EXPECT_NE(rejected_err.str().find("Unknown argument"), std::string::npos);
    EXPECT_THROW(ParseReconMvsArguments({"recon_mvs", "--input_path", "sparse", "--image_path", "images",
        "--output_path", root.string(), name, "1"}), std::invalid_argument);
  }
}

TEST_F(StageContractTest, DelaunayAcceptsDenseAndSparseInputTypes) {
  for (const auto* type : {"dense", "sparse", "SpArSe"}) {
    std::ostringstream out, err;
    EXPECT_EQ(RunDelaunayMesher({"delaunay_mesher", "--input_path", "absent",
        "--input_type", type, "--output_path", (root / "mesh.ply").string(),
        "--dry_run"}, out, err), 0) << err.str();
    EXPECT_NE(out.str().find("PLAN delaunay_mesher"), std::string::npos);
  }
  for (const auto& extra : {std::vector<std::string>{"--input_type", "invalid"},
        {"--model_path", "old-model"}, {"--input_type", "sparse", "--input_type", "dense"}}) {
    std::vector<std::string> args{"delaunay_mesher", "--input_path", "absent",
        "--output_path", (root / "mesh.ply").string(), "--resume", "--dry_run"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::ostringstream out, err;
    EXPECT_NE(RunDelaunayMesher(args, out, err), 0);
    EXPECT_TRUE(out.str().empty());
    EXPECT_FALSE(std::filesystem::exists(root / "mesh.ply"));
  }
}

TEST_F(StageContractTest, SurfaceStagesValidateBeforeDryRunOrResume) {
  using Runner = int (*)(const std::vector<std::string>&,
                         std::ostream&, std::ostream&);
  const std::vector<std::pair<Runner, std::vector<std::vector<std::string>>>> cases{
      {RunPoissonMesher, {{"--PoissonMeshing.unknown", "1"},
          {"--PoissonMeshing.depth", "-1"}, {"--PoissonMeshing.depth", "2.5"},
          {"--PoissonMeshing.color", "2"}, {"--PoissonMeshing.trim", "nan"},
          {"--PoissonMeshing.point_weight", "-1"},
          {"--PoissonMeshing.num_threads", "0"},
          {"--PoissonMeshing.depth=3", "--PoissonMeshing.depth", "4"},
          {"--PoissonMeshing.depth"}}},
      {RunDelaunayMesher, {{"--DelaunayMeshing.visibility_sigma", "-1"},
          {"--DelaunayMeshing.max_depth_dist", "2"},
          {"--DelaunayMeshing.max_proj_dist", "inf"}}},
      {RunMeshSimplifier, {{"--MeshSimplification.boundary_weight", "-1"},
          {"--MeshSimplification.num_threads", "2147483648"},
          {"--MeshSimplification.target_face_ratio", "2"},
          {"--MeshSimplification.interpolate_colors", "yes"}}},
      {RunMeshTexturer, {{"--Colite.texture_atlas_size", "-1"},
          {"--MeshTextureMapping.atlas_patch_padding", "2147483647"},
          {"--MeshTextureMapping.apply_color_correction", "2"}}}};
  for (const auto& [runner, invalid_options] : cases) {
    for (const auto& extra : invalid_options) {
      for (const bool resume : {false, true}) {
        std::vector<std::string> args{"surface", "--input_path", (root / "absent").string(),
            "--output_path", (root / "output/result.ply").string(), "--dry_run"};
        if (runner == RunMeshTexturer) {
          args.insert(args.end(), {"--workspace_path", root.string()});
        }
        if (resume) args.push_back("--resume");
        args.insert(args.end(), extra.begin(), extra.end());
        std::ostringstream out, err;
        EXPECT_NE(runner(args, out, err), 0) << extra.front();
        EXPECT_FALSE(err.str().empty());
        EXPECT_TRUE(out.str().empty()) << out.str();
        EXPECT_FALSE(std::filesystem::exists(root / "output"));
      }
    }
  }
}

TEST_F(StageContractTest, SurfaceStagesAcceptCanonicalValuesWithoutLoadingInputs) {
  using Runner = int (*)(const std::vector<std::string>&,
                         std::ostream&, std::ostream&);
  const std::vector<std::pair<Runner, std::vector<std::string>>> cases{
      {RunPoissonMesher, {"--PoissonMeshing.depth=3", "--PoissonMeshing.color", "true"}},
      {RunDelaunayMesher, {"--DelaunayMeshing.num_threads=-1"}},
      {RunMeshSimplifier, {"--MeshSimplification.boundary_weight=0",
                          "--MeshSimplification.interpolate_colors", "false"}},
      {RunMeshTexturer, {"--Colite.texture_atlas_size=16",
                        "--MeshTextureMapping.atlas_patch_padding=7",
                        "--MeshTextureMapping.apply_color_correction", "true"}}};
  for (const auto& [runner, extra] : cases) {
    std::vector<std::string> args{"surface", "--input_path", (root / "absent").string(),
        "--output_path", (root / "output/result.ply").string(), "--dry_run"};
    if (runner == RunMeshTexturer) {
      args.insert(args.end(), {"--workspace_path", root.string()});
    }
    args.insert(args.end(), extra.begin(), extra.end());
    std::ostringstream out, err;
    EXPECT_EQ(runner(args, out, err), 0) << err.str();
    EXPECT_FALSE(out.str().empty());
    EXPECT_FALSE(std::filesystem::exists(root / "output"));
  }
}

TEST_F(StageContractTest, MissingOrTruncatedDirectoryMemberInvalidatesResume) {
  const auto maps = root / "depth_maps";
  const StageSpec spec{"stereo", root, {}, {maps}, {"stereo", "--iterations", "1"}};
  int runs = 0;
  const auto runner = [&] {
    ++runs;
    Write(maps / "a.bin", "depth a");
    Write(maps / "nested/b.bin", "depth b");
    return 0;
  };
  std::ostringstream out, err;
  StageOptions options;
  ASSERT_EQ(RunStage(spec, options, runner, out, err), 0) << err.str();
  options.resume = true;
  ASSERT_EQ(RunStage(spec, options, runner, out, err), 0) << err.str();
  EXPECT_EQ(runs, 1);
  std::filesystem::remove(maps / "a.bin");
  ASSERT_EQ(RunStage(spec, options, runner, out, err), 0) << err.str();
  EXPECT_EQ(runs, 2);
  EXPECT_TRUE(std::filesystem::exists(maps / "a.bin"));
  Write(maps / "nested/b.bin", "");
  ASSERT_EQ(RunStage(spec, options, runner, out, err), 0) << err.str();
  EXPECT_EQ(runs, 3);
  EXPECT_EQ(std::filesystem::file_size(maps / "nested/b.bin"), 7);
}

TEST_F(StageContractTest, DoesNotCacheSuccessfulExitWithMissingOutputs) {
  const StageSpec spec{"stereo", root, {}, {root / "missing.bin"}, {"stereo"}};
  std::ostringstream out, err;
  StageOptions options;
  EXPECT_NE(RunStage(spec, options, [] { return 0; }, out, err), 0);
  EXPECT_NE(err.str().find("required outputs"), std::string::npos);
}

TEST_F(StageContractTest, MvsDryRunValidatesUnknownAndInvalidArguments) {
  using Runner = int (*)(const std::vector<std::string>&,
                         std::ostream&, std::ostream&);
  for (const Runner runner : {RunImageUndistorter, RunPatchMatchStereo,
                              RunStereoFusion}) {
    for (const auto& extra : {
             std::vector<std::string>{"--definitely_invalid", "1"},
             std::vector<std::string>{"--Colite.patch_match_previous_selection_probability_weight", "0.5"},
             std::vector<std::string>{"--num_threads", "0"}}) {
      std::vector<std::string> args{"stage", "--workspace_path", root.string(),
                                    "--dry_run"};
      args.insert(args.end(), extra.begin(), extra.end());
      std::ostringstream out, err;
      EXPECT_NE(runner(args, out, err), 0);
      EXPECT_EQ(out.str().find("PLAN "), std::string::npos);
      EXPECT_FALSE(std::filesystem::exists(root / ".colite"));
    }
  }
}

TEST_F(StageContractTest, MvsStagesShareEqualsAndDuplicateOptionSemantics) {
  using Runner = int (*)(const std::vector<std::string>&,
                         std::ostream&, std::ostream&);
  for (const Runner runner : {RunImageUndistorter, RunPatchMatchStereo,
                              RunStereoFusion}) {
    std::ostringstream out, err;
    const std::string threads = runner == RunImageUndistorter ? "num_threads" :
        (runner == RunPatchMatchStereo ? "PatchMatchStereo.num_threads" : "StereoFusion.num_threads");
    const std::vector<std::string> args{
        "stage", "--input_path=" + (root / "sparse").string(),
        "--image_path=" + (root / "images").string(),
        "--output_path=" + (root / "fused.ply").string(),
        "--" + threads + "=1", "--dry_run"};
    auto valid = args;
    if (runner == RunStereoFusion) valid.insert(valid.end(), {"--workspace_path", root.string()});
    ASSERT_EQ(runner(valid, out, err), 0) << err.str();
    EXPECT_NE(out.str().find("PLAN "), std::string::npos);
    auto duplicate = valid;
    duplicate.insert(duplicate.end(), {"--image_path", root.string()});
    EXPECT_NE(runner(duplicate, out, err), 0);
    EXPECT_NE(err.str().find("Duplicate option"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(root / ".colite"));
  }
}

TEST_F(StageContractTest, StandardMvsParametersControlDistinctStages) {
  const auto options = ParseReconMvsArguments({"recon_mvs",
      "--input_path", "sparse", "--image_path", "images", "--output_path", "result",
      "--Colite.run_patch_match", "true", "--PatchMatchStereo.num_iterations", "5",
      "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.geom_consistency_regularizer", "0.4",
      "--PatchMatchStereo.num_threads", "2", "--StereoFusion.num_threads", "3",
      "--PatchMatchStereo.max_image_size", "800", "--StereoFusion.max_image_size", "1200",
      "--PatchMatchStereo.depth_min", "2", "--PatchMatchStereo.depth_max", "6",
      "--PatchMatchStereo.window_step", "2"});
  EXPECT_EQ(options.patch_match_geometric_refinement_iterations, 5);
  EXPECT_FLOAT_EQ(options.patch_match_geometric_consistency_weight, 0.4F);
  EXPECT_EQ(options.patch_match_num_threads, 2);
  EXPECT_EQ(options.fusion_num_threads, 3);
  EXPECT_EQ(options.patch_match_max_image_size, 800);
  EXPECT_EQ(options.fusion_max_image_size, 1200);
  EXPECT_FLOAT_EQ(options.patch_match_depth_min, 2);
  EXPECT_FLOAT_EQ(options.patch_match_depth_max, 6);
  EXPECT_EQ(options.patch_match_window_step, 2);
  for (const std::string& name : {"--patch_match_iterations", "--fusion_min_num_observations",
                                 "--num_threads", "--patch_match_backend"}) {
    EXPECT_THROW(ParseReconMvsArguments({"recon_mvs", name, "1"}), std::invalid_argument);
  }
}

TEST_F(StageContractTest, MvsRejectsMalformedValuesAndInapplicableOptionsBeforeIO) {
  for (const auto& pair : {
           std::pair<std::string, std::string>{"--PatchMatchStereo.num_iterations", "5junk"},
           {"--PatchMatchStereo.num_iterations", "-1"},
           {"--PatchMatchStereo.num_iterations", "0"},
           {"--PatchMatchStereo.num_samples", "0"},
           {"--PatchMatchStereo.window_radius", "0"},
           {"--PatchMatchStereo.window_radius", "33"},
           {"--PatchMatchStereo.window_step", "3"},
           {"--PatchMatchStereo.num_iterations", "2147483648"},
           {"--PatchMatchStereo.num_iterations", "18446744073709551616"},
           {"--PatchMatchStereo.gpu_index", "0,1"},
           {"--PatchMatchStereo.num_threads", "-2"},
           {"--PatchMatchStereo.sigma_color", "nan"},
           {"--PatchMatchStereo.sigma_color", "0.2junk"},
           {"--PatchMatchStereo.filter", "2"},
           {"--StereoFusion.min_num_pixels", "5"}}) {
    std::ostringstream out, err;
    EXPECT_NE(RunPatchMatchStereo({"patch_match_stereo", "--workspace_path", root.string(),
                                   pair.first, pair.second, "--dry_run"}, out, err), 0);
    EXPECT_EQ(out.str().find("PLAN "), std::string::npos);
  }
}

TEST_F(StageContractTest, ReleaseMvsDefaultsUseColmapQualityParameters) {
  const auto stereo = ParseReconMvsArguments({"patch_match_stereo", "--workspace_path", root.string()}, MvsOptionScope::kPatchMatch);
  EXPECT_EQ(stereo.patch_match_iterations, 5);
  EXPECT_EQ(stereo.patch_match_window_radius, 5);
  EXPECT_EQ(stereo.patch_match_window_step, 1);
  EXPECT_EQ(stereo.patch_match_source_samples, 15);
  EXPECT_TRUE(stereo.patch_match_geom_consistency);
  EXPECT_EQ(stereo.patch_match_geometric_refinement_iterations, 5);
  EXPECT_TRUE(stereo.patch_match_filter);
  EXPECT_TRUE(stereo.patch_match_bilateral_ncc);
  EXPECT_TRUE(stereo.patch_match_directional_hmm);
  EXPECT_EQ(stereo.resolution_level, 0);
  EXPECT_EQ(stereo.source_selection_mode, "colmap");
  const auto fusion = ParseReconMvsArguments({"stereo_fusion", "--workspace_path", root.string(),
      "--output_path", (root / "fused.ply").string()}, MvsOptionScope::kFusion);
  EXPECT_EQ(fusion.fusion_min_num_observations, 5);
  EXPECT_EQ(fusion.fusion_max_num_observations, 10000);
  EXPECT_EQ(fusion.patch_match_geometric_refinement_iterations, 0);  // No hidden PatchMatch in Fusion.
  const auto zero = ParseReconMvsArguments({"patch_match_stereo", "--workspace_path", root.string(),
      "--PatchMatchStereo.filter_min_num_consistent", "0", "--PatchMatchStereo.geom_consistency_max_cost", "0",
      "--PatchMatchStereo.min_triangulation_angle", "45"}, MvsOptionScope::kPatchMatch);
  EXPECT_EQ(zero.patch_match_filter_min_num_consistent, 0);
  EXPECT_EQ(zero.patch_match_geom_consistency_max_cost, 0);
  EXPECT_THROW(ParseReconMvsArguments({"patch_match_stereo", "--workspace_path", root.string(),
      "--PatchMatchStereo.min_triangulation_angle", "180"}, MvsOptionScope::kPatchMatch), std::invalid_argument);
}

TEST_F(StageContractTest, ReconAutoValidatesSurfaceOptionsBeforeSfmIncludingDisabledStages) {
  for (const auto& extra : {
           std::vector<std::string>{"--PoissonMeshing.unknown", "1"},
           {"--PoissonMeshing.depth", "-1"},
           {"--PoissonMeshing.color", "2"},
           {"--DelaunayMeshing.visibility_sigma", "0"},
           {"--MeshSimplification.boundary_weight", "-1"},
           {"--Colite.texture_atlas_size", "-1"}}) {
    for (const bool dry_run : {false, true}) {
      std::vector<std::string> args{"recon_auto", "--image_path", "absent_images",
          "--output_path", (root / "output").string(), "--stop_after", "sfm"};
      args.insert(args.end(), extra.begin(), extra.end());
      if (dry_run) args.push_back("--dry_run");
      std::ostringstream out, err;
      EXPECT_NE(RunReconAuto(args, out, err), 0) << extra.front();
      EXPECT_TRUE(out.str().empty()) << out.str();
      EXPECT_FALSE(std::filesystem::exists(root / "output"));
    }
  }
  std::ostringstream out, err;
  EXPECT_EQ(RunReconAuto({"recon_auto", "--image_path", "absent_images",
      "--output_path", (root / "output").string(), "--PoissonMeshing.color=true",
      "--MeshSimplification.interpolate_colors=false", "--dry_run"}, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("PLAN poisson_mesher"), std::string::npos);
}

TEST_F(StageContractTest, PipelineSurfaceOptionsUseSharedValidationAndIndependentThreads) {
  const auto options = ParseReconMvsArguments({"recon_mvs", "--input_path", "sparse",
      "--image_path", "images", "--output_path", "result", "--PoissonMeshing.num_threads", "2",
      "--DelaunayMeshing.num_threads", "3", "--MeshTextureMapping.num_threads", "4"});
  EXPECT_EQ(options.PoissonOptions().num_threads, 2);
  EXPECT_EQ(options.DelaunayOptions().num_threads, 3);
  EXPECT_EQ(options.TexturingOptions().num_threads, 4);
  for (const auto& extra : {std::vector<std::string>{"--PoissonMeshing.num_threads", "0"},
           {"--DelaunayMeshing.num_threads", "-2"},
           {"--MeshTextureMapping.num_threads", "0"},
           {"--MeshTextureMapping.atlas_patch_padding", "2147483647"}}) {
    std::vector<std::string> args{"recon_mvs", "--input_path", "sparse",
        "--image_path", "images", "--output_path", "result"};
    args.insert(args.end(), extra.begin(), extra.end());
    EXPECT_THROW(ParseReconMvsArguments(args), std::invalid_argument) << extra.front();
  }
}

TEST_F(StageContractTest, ReconAutoValidatesAndRoutesMvsOptionsBeforeSfm) {
  std::ostringstream out, err;
  EXPECT_NE(RunReconAuto({"recon_auto", "--image_path", "images", "--output_path", root.string(),
      "--stop_after", "sfm", "--mvs.PatchMatchStereo.sigma_color", "bad", "--dry_run"}, out, err), 0);
  EXPECT_EQ(out.str().find("PLAN "), std::string::npos);
  out.str(""); err.str("");
  EXPECT_EQ(RunReconAuto({"recon_auto", "--image_path", "images", "--output_path", root.string(),
      "--mvs.PatchMatchStereo.num_threads", "2", "--mvs.StereoFusion.num_threads", "3",
      "--mvs.Colite.patch_match_backend", "cpu", "--dry_run"}, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("PLAN stereo_fusion"), std::string::npos);
  for (const auto scope : {MvsOptionScope::kUndistorter, MvsOptionScope::kPatchMatch, MvsOptionScope::kFusion}) {
    std::ostringstream help;
    PrintMvsOptionsHelp(scope, help);
    EXPECT_NE(help.str().find("default"), std::string::npos);
    const char* parameter = scope == MvsOptionScope::kUndistorter ? "--max_image_size" :
        (scope == MvsOptionScope::kPatchMatch ? "--PatchMatchStereo.num_iterations" : "--StereoFusion.min_num_pixels");
    EXPECT_NE(help.str().find(parameter), std::string::npos);
  }
}

}  // namespace
}  // namespace colite::exe
