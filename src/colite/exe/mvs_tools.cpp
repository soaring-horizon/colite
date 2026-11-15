#include "colite/exe/mvs_tools.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "colite/core/camera.h"
#include "colite/core/pose.h"
#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/mesh_texturer.h"
#include "colite/mvs/patch_match.h"
#include "colite/exe/stage.h"
#include "colite/exe/recon_mvs.h"
#include "colite/io/sfm_workspace.h"

namespace colite::exe {
namespace {

using Arguments = std::unordered_map<std::string, std::string>;

Arguments ParseArguments(const std::vector<std::string>& raw_args,
                         const std::vector<std::string>& algorithm_options) {
  const auto args = NormalizeMvsArguments(raw_args);
  std::set<std::string> allowed{"--input_path", "--output_path", "--stage_root",
      "--help", "--resume", "--force", "--dry_run"};
  allowed.insert(algorithm_options.begin(), algorithm_options.end());
  Arguments values;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (allowed.count(args[i]) == 0) {
      throw std::invalid_argument("Unknown argument: " + args[i]);
    }
    if (args[i] == "--help" || args[i] == "-h" ||
        args[i] == "--resume" || args[i] == "--force" ||
        args[i] == "--dry_run") {
      values[args[i] == "-h" ? "--help" : args[i]] = "1";
      continue;
    }
    if (args[i].rfind("--", 0) != 0 || i + 1 >= args.size() ||
        args[i + 1].rfind("--", 0) == 0) {
      throw std::invalid_argument("Expected --option value, got: " + args[i]);
    }
    const std::string option = args[i];
    values[option] = args[++i];
  }
  return values;
}

StageOptions ParseStageOptions(const Arguments& args) {
  StageOptions options;
  options.resume = args.count("--resume") != 0;
  options.force = args.count("--force") != 0;
  options.dry_run = args.count("--dry_run") != 0;
  if (options.resume && options.force) {
    throw std::invalid_argument("--resume and --force are mutually exclusive");
  }
  return options;
}

std::vector<std::string> AlgorithmArguments(
    const std::vector<std::string>& argv) {
  std::vector<std::string> result;
  for (const std::string& argument : argv) {
    if (argument != "--resume" && argument != "--force" &&
        argument != "--dry_run") {
      result.push_back(argument);
    }
  }
  result.push_back("@implementation=surface_tools_v11");
  return result;
}

std::filesystem::path StageRoot(const Arguments& args,
                                const std::filesystem::path& output,
                                bool output_is_directory = false) {
  if (args.count("--stage_root")) return args.at("--stage_root");
  return output_is_directory ? output : output.parent_path();
}

std::string Require(const Arguments& args, const std::string& name) {
  const auto it = args.find(name);
  if (it == args.end() || it->second.empty()) {
    throw std::invalid_argument("Missing required option: " + name);
  }
  return it->second;
}

std::filesystem::path ResolveModelPath(std::filesystem::path path) {
  if (!std::filesystem::exists(path / "cameras.txt") &&
      !std::filesystem::exists(path / "cameras.bin") &&
      std::filesystem::is_directory(path / "0")) {
    path /= "0";
  }
  return path;
}

template <typename T>
T Number(const Arguments& args, const std::string& name, T fallback) {
  const auto it = args.find(name);
  if (it == args.end()) return fallback;
  if constexpr (std::is_unsigned_v<T>) {
    if (it->second.find('-') != std::string::npos) {
      throw std::invalid_argument("Invalid value for " + name + ": " + it->second);
    }
  }
  std::istringstream stream(it->second);
  T value{};
  if (!(stream >> value) || !stream.eof()) {
    throw std::invalid_argument("Invalid value for " + name + ": " + it->second);
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Invalid value for " + name + ": " + it->second);
    }
  }
  return value;
}

bool Boolean(const Arguments& args, const std::string& name, bool fallback) {
  const auto it = args.find(name);
  if (it == args.end()) return fallback;
  if (it->second == "1" || it->second == "true") return true;
  if (it->second == "0" || it->second == "false") return false;
  throw std::invalid_argument("Invalid boolean for " + name + ": " + it->second);
}

std::string DelaunayInputType(const Arguments& args) {
  std::string type = args.count("--input_type") ? args.at("--input_type") : "dense";
  std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (type != "dense" && type != "sparse") {
    throw std::invalid_argument("--input_type must be dense or sparse");
  }
  return type;
}

template <typename Options>
void ValidateOptions(const Options& options, const std::string& command) {
  if (!options.Check()) {
    throw std::invalid_argument("Invalid " + command + " options");
  }
}

template <typename Options>
Options ParseOptions(const Arguments& args);

template <>
mvs::PoissonMeshingOptions ParseOptions<mvs::PoissonMeshingOptions>(
    const Arguments& args) {
  mvs::PoissonMeshingOptions options;
  options.depth = Number(args, "--PoissonMeshing.depth", options.depth);
  options.point_weight = Number(args, "--PoissonMeshing.point_weight", options.point_weight);
  options.color = Boolean(args, "--PoissonMeshing.color", options.color);
  options.trim = Number(args, "--PoissonMeshing.trim", options.trim);
  options.num_threads = Number(args, "--PoissonMeshing.num_threads", options.num_threads);
  ValidateOptions(options, "PoissonMeshing");
  return options;
}

template <>
mvs::MeshSimplificationOptions ParseOptions<mvs::MeshSimplificationOptions>(
    const Arguments& args) {
  mvs::MeshSimplificationOptions options;
  options.target_face_ratio = Number(args, "--MeshSimplification.target_face_ratio", options.target_face_ratio);
  options.max_error = Number(args, "--MeshSimplification.max_error", options.max_error);
  options.boundary_weight = Number(args, "--MeshSimplification.boundary_weight", options.boundary_weight);
  options.interpolate_colors = Boolean(args, "--MeshSimplification.interpolate_colors", options.interpolate_colors);
  options.num_threads = Number(args, "--MeshSimplification.num_threads", options.num_threads);
  ValidateOptions(options, "MeshSimplification");
  return options;
}

template <>
mvs::DelaunayMeshingOptions ParseOptions<mvs::DelaunayMeshingOptions>(
    const Arguments& args) {
  mvs::DelaunayMeshingOptions options;
  options.max_proj_dist = Number(args, "--DelaunayMeshing.max_proj_dist", options.max_proj_dist);
  options.max_depth_dist = Number(args, "--DelaunayMeshing.max_depth_dist", options.max_depth_dist);
  options.visibility_sigma = Number(args, "--DelaunayMeshing.visibility_sigma", options.visibility_sigma);
  options.distance_sigma_factor = Number(args, "--DelaunayMeshing.distance_sigma_factor", options.distance_sigma_factor);
  options.quality_regularization = Number(args, "--DelaunayMeshing.quality_regularization", options.quality_regularization);
  options.max_side_length_factor = Number(args, "--DelaunayMeshing.max_side_length_factor", options.max_side_length_factor);
  options.max_side_length_percentile = Number(args, "--DelaunayMeshing.max_side_length_percentile", options.max_side_length_percentile);
  options.num_threads = Number(args, "--DelaunayMeshing.num_threads", options.num_threads);
  ValidateOptions(options, "DelaunayMeshing");
  return options;
}

template <>
mvs::MeshTexturingOptions ParseOptions<mvs::MeshTexturingOptions>(
    const Arguments& args) {
  mvs::MeshTexturingOptions options;
  options.atlas_size = Number(args, "--Colite.texture_atlas_size", options.atlas_size);
  options.max_atlas_size = Number(args, "--Colite.texture_max_atlas_size", options.max_atlas_size);
  options.atlas_patch_padding = Number(args, "--MeshTextureMapping.atlas_patch_padding", options.atlas_patch_padding);
  options.inpaint_radius = Number(args, "--MeshTextureMapping.inpaint_radius", options.inpaint_radius);
  options.texture_scale = Number(args, "--MeshTextureMapping.texture_scale_factor", options.texture_scale);
  options.min_cos_normal_angle = Number(args, "--MeshTextureMapping.min_cos_normal_angle", options.min_cos_normal_angle);
  options.min_visible_vertices = Number(args, "--MeshTextureMapping.min_visible_vertices", options.min_visible_vertices);
  options.view_selection_smoothing_iterations = Number(args, "--MeshTextureMapping.view_selection_smoothing_iterations", options.view_selection_smoothing_iterations);
  options.view_selection_min_component_size = Number(
      args, "--Colite.texture_view_selection_min_component_size",
      options.view_selection_min_component_size);
  options.visibility_resolution = Number(
      args, "--Colite.texture_visibility_resolution",
      options.visibility_resolution);
  options.occlusion_relative_tolerance = Number(
      args, "--Colite.texture_occlusion_relative_tolerance",
      options.occlusion_relative_tolerance);
  options.use_vertex_color_fallback = Boolean(
      args, "--Colite.texture_use_vertex_color_fallback",
      options.use_vertex_color_fallback);
  options.apply_color_correction = Boolean(args, "--MeshTextureMapping.apply_color_correction", options.apply_color_correction);
  options.color_correction_regularization = Number(args, "--MeshTextureMapping.color_correction_regularization", options.color_correction_regularization);
  options.num_threads = Number(args, "--MeshTextureMapping.num_threads", options.num_threads);
  ValidateOptions(options, "MeshTextureMapping");
  return options;
}

struct SurfaceTool {
  const char* name;
  std::vector<std::string> options;
  void (*validate)(const Arguments&);
};

const SurfaceTool& FindSurfaceTool(const std::string& name) {
  static const std::vector<SurfaceTool> tools{
      {"poisson_mesher", {"--PoissonMeshing.depth",
        "--PoissonMeshing.point_weight", "--PoissonMeshing.color",
        "--PoissonMeshing.trim", "--PoissonMeshing.num_threads"}, [](const Arguments& args) {
         (void)ParseOptions<mvs::PoissonMeshingOptions>(args);
      }},
      {"mesh_simplifier", {"--MeshSimplification.target_face_ratio", "--MeshSimplification.max_error",
        "--MeshSimplification.boundary_weight", "--MeshSimplification.interpolate_colors",
        "--MeshSimplification.num_threads"}, [](const Arguments& args) {
         (void)ParseOptions<mvs::MeshSimplificationOptions>(args);
      }},
      {"delaunay_mesher", {"--input_type",
        "--DelaunayMeshing.max_proj_dist", "--DelaunayMeshing.max_depth_dist",
        "--DelaunayMeshing.visibility_sigma", "--DelaunayMeshing.distance_sigma_factor",
        "--DelaunayMeshing.quality_regularization", "--DelaunayMeshing.max_side_length_factor",
        "--DelaunayMeshing.max_side_length_percentile", "--DelaunayMeshing.num_threads"}, [](const Arguments& args) {
         (void)DelaunayInputType(args);
         (void)ParseOptions<mvs::DelaunayMeshingOptions>(args);
      }},
      {"mesh_texturer", {"--workspace_path", "--output_type", "--Colite.texture_write_obj",
        "--Colite.texture_atlas_size", "--Colite.texture_max_atlas_size",
        "--MeshTextureMapping.atlas_patch_padding", "--MeshTextureMapping.inpaint_radius",
        "--MeshTextureMapping.texture_scale_factor", "--MeshTextureMapping.min_cos_normal_angle",
        "--MeshTextureMapping.min_visible_vertices", "--MeshTextureMapping.view_selection_smoothing_iterations",
        "--Colite.texture_view_selection_min_component_size", "--Colite.texture_visibility_resolution",
        "--Colite.texture_occlusion_relative_tolerance", "--Colite.texture_use_vertex_color_fallback",
        "--MeshTextureMapping.apply_color_correction", "--MeshTextureMapping.color_correction_regularization",
        "--MeshTextureMapping.num_threads"}, [](const Arguments& args) {
         (void)ParseOptions<mvs::MeshTexturingOptions>(args);
      }}};
  for (const auto& tool : tools) {
    if (tool.name == name) return tool;
  }
  throw std::invalid_argument("Unknown surface tool: " + name);
}

Arguments ParseToolArguments(const std::vector<std::string>& args,
                             const std::string& tool) {
  return ParseArguments(args, FindSurfaceTool(tool).options);
}

struct ModelImage {
  std::string name;
  mvs::PatchMatchView view;
};

std::vector<ModelImage> ReadModelImages(const std::filesystem::path& model_path,
                                        bool load_pixels,
                                        const std::filesystem::path& image_path,
                                        std::vector<mvs::MeshTextureView>* textures) {
  std::vector<ImageId> model_order;
  const Reconstruction reconstruction = ReadColmapReconstruction(
      model_path, ColmapModelFormat::kAuto, &model_order);
  std::vector<ModelImage> images;
  images.reserve(reconstruction.RegisteredImageIds().size());
  for (const ImageId image_id : model_order) {
    const Image& source = reconstruction.GetImage(image_id);
    if (!source.cam_from_world.has_value()) {
      throw std::runtime_error("Registered model image has no pose");
    }
    ModelImage image;
    image.name = source.name;
    image.view.image_id = source.id;
    image.view.camera = reconstruction.GetCamera(source.camera_id);
    image.view.cam_from_world = *source.cam_from_world;
    images.push_back(std::move(image));
  }
  for (const ModelImage& image : images) {
    if (load_pixels) {
      textures->push_back(mvs::ReadMeshTextureView(image.view, image_path / image.name));
    }
  }
  return images;
}

template <typename Function>
int GuardedRun(Function&& function, std::ostream& err) {
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << '\n';
    return 1;
  }
}

}  // namespace

void ValidateSurfaceArguments(const std::string& tool,
                               const std::vector<std::string>& argv) {
  const Arguments args = ParseToolArguments(argv, tool);
  if (args.count("--help")) return;
  (void)Require(args, "--input_path");
  (void)Require(args, "--output_path");
  if (tool == "mesh_texturer") {
    (void)Require(args, "--workspace_path");
    if (args.count("--output_type")) (void)mvs::TexturedMeshOutputTypeFromString(args.at("--output_type"));
    (void)Boolean(args, "--Colite.texture_write_obj", false);
  }
  (void)ParseStageOptions(args);
  FindSurfaceTool(tool).validate(args);
}

int RunPoissonMesher(const std::vector<std::string>& argv,
                     std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseToolArguments(argv, "poisson_mesher");
    if (args.count("--help")) {
      out << "Usage: poisson_mesher --input_path fused.ply --output_path mesh.ply "
             "[--PoissonMeshing.depth 13] [--PoissonMeshing.point_weight 1] "
             "[--PoissonMeshing.color 1] [--PoissonMeshing.trim 10] "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path input = Require(args, "--input_path");
    const std::filesystem::path output = Require(args, "--output_path");
    const auto options = ParseOptions<mvs::PoissonMeshingOptions>(args);
    StageSpec spec{"poisson_mesher", StageRoot(args, output), {input},
                   {output}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      if (!mvs::PoissonMeshing(options, input, output)) {
        throw std::runtime_error("Poisson reconstruction failed");
      }
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunMeshSimplifier(const std::vector<std::string>& argv,
                      std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseToolArguments(argv, "mesh_simplifier");
    if (args.count("--help")) {
      out << "Usage: mesh_simplifier --input_path mesh.ply --output_path simplified.ply "
             "[--MeshSimplification.target_face_ratio 0.1] "
             "[--MeshSimplification.max_error 0] [--MeshSimplification.boundary_weight 1000] "
             "[--MeshSimplification.interpolate_colors 1] [--MeshSimplification.num_threads -1] "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path input_path = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    const auto options = ParseOptions<mvs::MeshSimplificationOptions>(args);
    StageSpec spec{"mesh_simplifier", StageRoot(args, output_path),
                   {input_path}, {output_path}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      const mvs::DenseTriangleMesh input =
          mvs::ReadDenseTriangleMeshPly(input_path);
      mvs::WriteDenseTriangleMeshPly(output_path,
                                     mvs::SimplifyMesh(input, options));
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunDelaunayMesher(const std::vector<std::string>& argv,
                      std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseToolArguments(argv, "delaunay_mesher");
    if (args.count("--help")) {
      out << "Usage: delaunay_mesher --input_path dense_workspace --output_path mesh.ply "
             "[--input_type dense|sparse] [--resume|--force] [--dry_run]\n"
             "Default input_type=dense: input_path contains sparse/, fused.ply and fused.ply.vis.\n"
             "input_type=sparse: input_path is a COLMAP sparse model (BIN or TXT).\n"
             "DelaunayMeshing options: max_proj_dist=20, max_depth_dist=0.05, visibility_sigma=3, "
             "distance_sigma_factor=1, quality_regularization=1, max_side_length_factor=25, "
             "max_side_length_percentile=95, num_threads=-1.\n";
      return;
    }
    const std::filesystem::path workspace = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    const bool sparse = DelaunayInputType(args) == "sparse";
    const std::filesystem::path model = ResolveModelPath(sparse ? workspace : workspace / "sparse");
    const auto visibility_path = workspace / "fused.ply.vis";
    const auto options = ParseOptions<mvs::DelaunayMeshingOptions>(args);
    StageSpec spec{"delaunay_mesher", StageRoot(args, output_path),
                   sparse ? std::vector<std::filesystem::path>{model}
                          : std::vector<std::filesystem::path>{workspace / "fused.ply", visibility_path, model},
                   {output_path}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      if (sparse) {
        mvs::WriteDenseTriangleMeshPly(output_path,
            mvs::SparseDelaunayMeshing(options, ReadColmapReconstruction(model)));
        return 0;
      }
      const mvs::DenseTriangleMesh cloud =
          mvs::ReadDenseTriangleMeshPly(workspace / "fused.ply");
      const auto visibility =
          mvs::ReadPointsVisibilityColmapBinary(visibility_path);
      std::vector<mvs::MeshTextureView> unused;
      const auto model_images = ReadModelImages(model, false, {}, &unused);
      std::vector<mvs::PatchMatchView> views;
      views.reserve(model_images.size());
      for (const ModelImage& image : model_images) views.push_back(image.view);
      mvs::WriteDenseTriangleMeshPly(
          output_path,
          mvs::DelaunayMeshing(options, cloud.vertices, visibility, views));
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunMeshTexturer(const std::vector<std::string>& argv,
                    std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseToolArguments(argv, "mesh_texturer");
    if (args.count("--help")) {
      out << "Usage: mesh_texturer --workspace_path workspace --input_path mesh.ply --output_path output_dir "
             "[--output_type BIN|TXT] [--Colite.texture_write_obj 0|1] "
             "[--resume|--force] [--dry_run]\n"
             "Outputs: mesh.ply and texture.png; default output_type BIN.\n"
             "MeshTextureMapping options: min_cos_normal_angle=0.1, min_visible_vertices=3, "
             "view_selection_smoothing_iterations=3, atlas_patch_padding=2, inpaint_radius=5, "
             "texture_scale_factor=1, apply_color_correction=1, color_correction_regularization=0.1, num_threads=-1.\n"
             "Colite extensions: texture_atlas_size=4096, texture_max_atlas_size=32768, "
             "texture_view_selection_min_component_size=64, texture_visibility_resolution=1024, "
             "texture_occlusion_relative_tolerance=0.01, texture_use_vertex_color_fallback=0.\n";
      return;
    }
    const std::filesystem::path input_path = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    const auto workspace = std::filesystem::path(Require(args, "--workspace_path"));
    const auto model_path = ResolveModelPath(workspace / "sparse");
    const auto image_path = workspace / "images";
    const auto output_type = mvs::TexturedMeshOutputTypeFromString(
        args.count("--output_type") ? args.at("--output_type") : "BIN");
    const bool write_obj = Boolean(args, "--Colite.texture_write_obj", false);
    const auto options = ParseOptions<mvs::MeshTexturingOptions>(args);
    StageSpec spec{"mesh_texturer", StageRoot(args, output_path, true),
                   {input_path, model_path, image_path},
                   {output_path / "mesh.ply", output_path / "texture.png"},
                   AlgorithmArguments(argv)};
    if (write_obj) {
      spec.outputs.push_back(output_path / "mesh.obj");
      spec.outputs.push_back(output_path / "mesh.mtl");
    }
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      std::vector<mvs::MeshTextureView> views;
      ReadModelImages(model_path, true, image_path, &views);
      const mvs::DenseTriangleMesh mesh =
          mvs::ReadDenseTriangleMeshPly(input_path);
      mvs::WriteTexturedTriangleMesh(output_path,
                                     mvs::TextureMesh(mesh, views, options), output_type, write_obj);
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

}  // namespace colite::exe
