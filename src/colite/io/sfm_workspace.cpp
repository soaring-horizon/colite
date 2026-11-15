#include "colite/io/sfm_workspace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "colite/core/geometry.h"

namespace colite {
namespace {

constexpr std::uint32_t kFeatureFileMagic = 0x434C4654U;
constexpr std::uint32_t kFeatureFileVersion = 3;
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kMaxFeatureFilePayloadBytes = 8ULL * 1024 * 1024 * 1024;

void HashBytes(std::uint64_t* hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t idx = 0; idx < size; ++idx) {
    *hash ^= bytes[idx];
    *hash *= kFnvPrime;
  }
}

std::uint64_t FeaturePayloadChecksum(const FeatureSet& features,
                                     bool include_source_shapes = false) {
  std::uint64_t hash = kFnvOffsetBasis;
  if (include_source_shapes) {
    HashBytes(&hash, &features.source_keypoint_columns, sizeof(std::uint32_t));
  }
  for (const FeatureKeypoint& keypoint : features.keypoints) {
    HashBytes(&hash, &keypoint.x, sizeof(keypoint.x));
    HashBytes(&hash, &keypoint.y, sizeof(keypoint.y));
    HashBytes(&hash, &keypoint.scale, sizeof(keypoint.scale));
    HashBytes(&hash, &keypoint.orientation, sizeof(keypoint.orientation));
    HashBytes(&hash, &keypoint.response, sizeof(keypoint.response));
  }
  if (include_source_shapes && features.source_keypoint_shapes.size() > 0) {
    HashBytes(&hash, features.source_keypoint_shapes.data(),
              static_cast<std::size_t>(features.source_keypoint_shapes.size()) * sizeof(float));
  }
  if (features.descriptors.size() > 0) {
    HashBytes(&hash, features.descriptors.data(),
              static_cast<std::size_t>(features.descriptors.size()) *
                  sizeof(float));
  }
  return hash;
}

class FileDescriptorSource final : public DescriptorSource {
 public:
  FileDescriptorSource(std::filesystem::path path, Eigen::Index rows,
                       Eigen::Index cols, std::streamoff offset,
                       std::uintmax_t bytes,
                       std::filesystem::file_time_type write_time)
      : path_(std::move(path)), rows_(rows), cols_(cols), offset_(offset),
        bytes_(bytes), write_time_(write_time) {}

  Eigen::Index Rows() const override { return rows_; }
  Eigen::Index Cols() const override { return cols_; }

  DescriptorMatrix ReadRows(
      const std::vector<Point2DIdx>& indices) const override {
    DescriptorMatrix result(static_cast<Eigen::Index>(indices.size()), cols_);
    std::vector<std::pair<Point2DIdx, std::size_t>> ordered;
    ordered.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= static_cast<std::uint64_t>(rows_)) {
        throw std::out_of_range("Descriptor row index is out of range");
      }
      ordered.emplace_back(indices[i], i);
    }
    if (indices.empty() || cols_ == 0) return result;
    CheckUnchanged();
    std::ifstream file(path_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot reopen descriptors: " + path_.string());
    std::sort(ordered.begin(), ordered.end());
    // Merge nearby requests into bounded reads, avoiding one filesystem seek
    // per descriptor while retaining only the requested rows in the result.
    const std::size_t row_bytes = static_cast<std::size_t>(cols_) * sizeof(float);
    const std::size_t block_rows = std::max<std::size_t>(1, 64 * 1024 / row_bytes);
    std::vector<float> buffer(block_rows * static_cast<std::size_t>(cols_));
    for (std::size_t first = 0; first < ordered.size();) {
      std::size_t end = first + 1;
      while (end < ordered.size() &&
             ordered[end].first - ordered[first].first < block_rows) ++end;
      const std::size_t count = ordered[end - 1].first - ordered[first].first + 1;
      file.seekg(offset_ + static_cast<std::streamoff>(ordered[first].first) * row_bytes);
      file.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(count * row_bytes));
      if (!file) throw std::runtime_error("Truncated deferred descriptors: " + path_.string());
      for (std::size_t i = first; i < end; ++i) {
        std::memcpy(result.row(static_cast<Eigen::Index>(ordered[i].second)).data(),
                    buffer.data() + (ordered[i].first - ordered[first].first) * cols_, row_bytes);
      }
      first = end;
    }
    CheckUnchanged();
    return result;
  }

 private:
  void CheckUnchanged() const {
    if (std::filesystem::file_size(path_) != bytes_ ||
        std::filesystem::last_write_time(path_) != write_time_) {
      throw std::runtime_error("Deferred feature file changed: " + path_.string());
    }
  }
  std::filesystem::path path_;
  Eigen::Index rows_, cols_;
  std::streamoff offset_;
  std::uintmax_t bytes_;
  std::filesystem::file_time_type write_time_;
};

FeatureExtractorType ParseFeatureExtractorType(std::uint32_t value) {
  const auto type = static_cast<FeatureExtractorType>(value);
  if (type != FeatureExtractorType::kSift &&
      type != FeatureExtractorType::kAlikedN16Rot &&
      type != FeatureExtractorType::kAlikedN32) {
    throw std::runtime_error("Unsupported feature extractor type");
  }
  return type;
}

DescriptorScalarType ParseDescriptorScalarType(std::uint32_t value) {
  const auto type = static_cast<DescriptorScalarType>(value);
  if (type != DescriptorScalarType::kFloat32) {
    throw std::runtime_error("Unsupported descriptor scalar type");
  }
  return type;
}

DescriptorMetric ParseDescriptorMetric(std::uint32_t value) {
  const auto metric = static_cast<DescriptorMetric>(value);
  if (metric != DescriptorMetric::kSiftAngular &&
      metric != DescriptorMetric::kCosine) {
    throw std::runtime_error("Unsupported descriptor metric");
  }
  return metric;
}

DescriptorNormalization ParseDescriptorNormalization(std::uint32_t value) {
  const auto normalization = static_cast<DescriptorNormalization>(value);
  if (normalization != DescriptorNormalization::kNone &&
      normalization != DescriptorNormalization::kRootL1 &&
      normalization != DescriptorNormalization::kL2) {
    throw std::runtime_error("Unsupported descriptor normalization");
  }
  return normalization;
}

std::vector<std::string> DataLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open workspace file: " +
                             path.string());
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::pair<std::string, std::string>> ColmapImageEntries(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP model file: " +
                             path.string());
  }
  std::vector<std::pair<std::string, std::string>> entries;
  std::string header;
  while (std::getline(file, header)) {
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header.empty() || header.front() == '#') {
      continue;
    }
    std::string points;
    if (!std::getline(file, points)) {
      throw std::runtime_error(
          "COLMAP images.txt is missing the POINTS2D line after: " + header);
    }
    if (!points.empty() && points.back() == '\r') points.pop_back();
    if (!points.empty() && points.front() == '#') {
      throw std::runtime_error(
          "COLMAP images.txt has a comment where a POINTS2D line is expected");
    }
    entries.emplace_back(std::move(header), std::move(points));
  }
  return entries;
}

bool CameraParametersEqual(const Camera& lhs, const Camera& rhs) {
  if (lhs.model_id != rhs.model_id || lhs.width != rhs.width ||
      lhs.height != rhs.height || lhs.params.size() != rhs.params.size()) {
    return false;
  }
  for (std::size_t idx = 0; idx < lhs.params.size(); ++idx) {
    const double scale =
        std::max({1.0, std::abs(lhs.params[idx]), std::abs(rhs.params[idx])});
    if (std::abs(lhs.params[idx] - rhs.params[idx]) > 1e-10 * scale) {
      return false;
    }
  }
  return true;
}

std::string Point3DIdForText(Point3DId point3d_id) {
  return point3d_id == kInvalidPoint3DId ? "-1" : std::to_string(point3d_id);
}

Point3DId ParsePoint3DId(const std::string& text) {
  if (text == "-1") return kInvalidPoint3DId;
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::runtime_error("Invalid COLMAP POINT3D_ID: " + text);
  }
  return std::stoull(text);
}

int GeometryTypeToInt(TwoViewGeometry::Type type) {
  return static_cast<int>(type);
}

TwoViewGeometry::Type GeometryTypeFromInt(int value) {
  if (value < 0 || value > 9) throw std::runtime_error("Invalid two-view geometry type");
  return static_cast<TwoViewGeometry::Type>(value);
}

std::string VerifiedPrefix(const ImagePair& pair, const TwoViewGeometry& geometry) {
  const Pose3 pose = geometry.cam2_from_cam1.value_or(Pose3{});
  std::ostringstream prefix;
  prefix << pair.image_id1 << "\t" << pair.image_id2 << "\t"
         << static_cast<int>(geometry.type) << "\t" << std::setprecision(17)
         << geometry.median_triangulation_angle << "\t" << pose.q_cw.w()
         << "\t" << pose.q_cw.x() << "\t" << pose.q_cw.y() << "\t" << pose.q_cw.z()
         << "\t" << pose.t_cw.x() << "\t" << pose.t_cw.y() << "\t" << pose.t_cw.z() << "\t";
  return prefix.str();
}

void ValidateVerifiedGeometry(const TwoViewGeometry& geometry) {
  GeometryTypeFromInt(static_cast<int>(geometry.type));
  if (!std::isfinite(geometry.median_triangulation_angle)) {
    throw std::runtime_error("Non-finite triangulation angle");
  }
  for (const auto* matrix : {&geometry.F, &geometry.E, &geometry.H}) {
    if (*matrix && !(*matrix)->allFinite()) throw std::runtime_error("Non-finite geometry matrix");
  }
  if (geometry.cam2_from_cam1 &&
      (!geometry.cam2_from_cam1->q_cw.coeffs().allFinite() ||
       geometry.cam2_from_cam1->q_cw.norm() <= 1e-12 ||
       !geometry.cam2_from_cam1->t_cw.allFinite())) {
    throw std::runtime_error("Invalid relative pose");
  }
}

template <typename T>
void WriteBinary(std::ofstream* file, const T& value) {
  file->write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!*file) {
    throw std::runtime_error("Failed to write binary file");
  }
}

template <typename T>
void ReadBinary(std::ifstream* file, T* value) {
  file->read(reinterpret_cast<char*>(value), sizeof(T));
  if (!*file) {
    throw std::runtime_error("Unexpected end of binary workspace file");
  }
}

bool HostIsLittleEndian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

// COLMAP model files have a little-endian wire format, independent of the host.
template <typename T>
void WriteColmapValue(std::ofstream* file, const T& value) {
  std::array<char, sizeof(T)> bytes;
  std::memcpy(bytes.data(), &value, sizeof(T));
  if (!HostIsLittleEndian()) std::reverse(bytes.begin(), bytes.end());
  file->write(bytes.data(), bytes.size());
  if (!*file) throw std::runtime_error("Failed to write COLMAP binary model");
}

template <typename T>
void ReadColmapValue(std::ifstream* file, T* value) {
  std::array<char, sizeof(T)> bytes;
  file->read(bytes.data(), bytes.size());
  if (!*file) throw std::runtime_error("Unexpected end of COLMAP binary model");
  if (!HostIsLittleEndian()) std::reverse(bytes.begin(), bytes.end());
  std::memcpy(value, bytes.data(), sizeof(T));
}

std::string ReadNullTerminated(std::ifstream* file) {
  std::string value;
  char character = '\0';
  while (file->get(character)) {
    if (character == '\0') {
      return value;
    }
    value.push_back(character);
  }
  throw std::runtime_error("Unexpected end of COLMAP binary image name");
}

template <typename Id, typename Map>
std::vector<Id> SortedIds(const Map& values) {
  std::vector<Id> ids;
  ids.reserve(values.size());
  for (const auto& [id, value] : values) {
    (void)value;
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

bool HasCompleteModel(const std::filesystem::path& path,
                      const std::string& extension) {
  return std::filesystem::is_regular_file(path / ("cameras." + extension)) &&
         std::filesystem::is_regular_file(path / ("images." + extension)) &&
         std::filesystem::is_regular_file(path / ("points3D." + extension));
}

std::filesystem::path NormalizeRequiredFile(
    const std::filesystem::path& path) {
  return path.lexically_normal();
}

std::string ToHex(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

std::string CacheBodyChecksum(const std::string& body) {
  std::uint64_t hash = kFnvOffsetBasis;
  HashBytes(&hash, body.data(), body.size());
  return ToHex(hash);
}

std::string CacheArtifactFingerprint(
    const std::filesystem::path& workspace,
    const std::vector<std::filesystem::path>& required_files) {
  std::uint64_t hash = kFnvOffsetBasis;
  const auto mix = [&](const std::string& value) {
    const std::uint64_t size = value.size();
    HashBytes(&hash, &size, sizeof(size));
    HashBytes(&hash, value.data(), value.size());
  };
  for (const auto& required : required_files) {
    const auto root = (workspace / required).lexically_normal();
    std::error_code error;
    if (!std::filesystem::exists(root, error) || error) return {};
    std::vector<std::filesystem::path> paths{root};
    if (std::filesystem::is_directory(root, error)) {
      for (std::filesystem::recursive_directory_iterator it(root, error), end;
           !error && it != end; it.increment(error)) {
        paths.push_back(it->path());
      }
      if (error) return {};
    }
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
      mix(path.lexically_normal().lexically_relative(workspace.lexically_normal()).generic_string());
      if (std::filesystem::is_regular_file(path, error)) {
        mix("file");
        const auto size = std::filesystem::file_size(path, error);
        if (error) return {};
        mix(std::to_string(size));
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) return {};
        mix(std::to_string(modified.time_since_epoch().count()));
      } else if (!error && std::filesystem::is_directory(path, error)) {
        mix("directory");
      } else {
        return {};
      }
      if (error) return {};
    }
  }
  return ToHex(hash);
}

}  // namespace

ColmapModelFormat ColmapModelFormatFromString(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (normalized == "auto") return ColmapModelFormat::kAuto;
  if (normalized == "txt" || normalized == "text") return ColmapModelFormat::kText;
  if (normalized == "bin" || normalized == "binary") return ColmapModelFormat::kBinary;
  if (normalized == "both" || normalized == "txt+bin") {
    return ColmapModelFormat::kTextAndBinary;
  }
  throw std::invalid_argument(
      "COLMAP model format must be auto, txt, bin, or both: " + value);
}

std::string ColmapModelFormatToString(const ColmapModelFormat format) {
  switch (format) {
    case ColmapModelFormat::kAuto: return "auto";
    case ColmapModelFormat::kText: return "txt";
    case ColmapModelFormat::kBinary: return "bin";
    case ColmapModelFormat::kTextAndBinary: return "both";
  }
  throw std::invalid_argument("Unknown COLMAP model format");
}

ColmapModelFormat DetectColmapModelFormat(
    const std::filesystem::path& model_path) {
  if (HasCompleteModel(model_path, "bin")) {
    return ColmapModelFormat::kBinary;
  }
  if (HasCompleteModel(model_path, "txt")) {
    return ColmapModelFormat::kText;
  }
  throw std::runtime_error(
      "COLMAP model requires a complete cameras/images/points3D .bin or .txt "
      "file set: " + model_path.string());
}

std::string ColmapModelFingerprint(const std::filesystem::path& model_path,
                                   ColmapModelFormat format) {
  if (format == ColmapModelFormat::kAuto) format = DetectColmapModelFormat(model_path);
  if (format != ColmapModelFormat::kText && format != ColmapModelFormat::kBinary) {
    throw std::invalid_argument("Input model fingerprint requires auto, txt, or bin");
  }
  const std::string extension = format == ColmapModelFormat::kBinary ? ".bin" : ".txt";
  std::uint64_t hash = kFnvOffsetBasis;
  std::array<char, 64 * 1024> buffer;
  for (const std::string name : {"cameras", "images", "points3D", "rigs", "frames"}) {
    const std::string filename = name + extension;
    HashBytes(&hash, filename.data(), filename.size());
    const auto path = model_path / filename;
    const bool optional = name == "rigs" || name == "frames";
    const bool exists = std::filesystem::exists(path);
    HashBytes(&hash, &exists, sizeof(exists));
    if (optional && !exists) continue;
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot fingerprint COLMAP model file: " + path.string());
    const auto size = std::filesystem::file_size(path);
    HashBytes(&hash, &size, sizeof(size));
    while (file) {
      file.read(buffer.data(), buffer.size());
      HashBytes(&hash, buffer.data(), static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof()) throw std::runtime_error("Failed to fingerprint COLMAP model file: " + path.string());
  }
  return ToHex(hash);
}

std::filesystem::path WorkspaceFeaturesPath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "features";
}

std::filesystem::path WorkspaceMatchesPath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "matches";
}

std::filesystem::path WorkspaceSparsePath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "sparse";
}

std::filesystem::path WorkspaceCacheMetadataPath(
    const std::filesystem::path& workspace_path,
    const std::string& stage) {
  return workspace_path / "cache" / (stage + ".txt");
}

std::string StableConfigHash(
    const std::vector<std::pair<std::string, std::string>>& entries) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&](const char ch) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  };

  for (const auto& [key, value] : entries) {
    for (const char ch : key) {
      mix(ch);
    }
    mix('=');
    for (const char ch : value) {
      mix(ch);
    }
    mix('\n');
  }
  return ToHex(hash);
}

void WriteWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage,
    const std::string& config_hash,
    const std::map<std::string, std::string>& values,
    const std::vector<std::filesystem::path>& required_files) {
  const auto path = WorkspaceCacheMetadataPath(workspace_path, stage);
  std::filesystem::create_directories(path.parent_path());
  std::ostringstream body;
  body << "schema_version\t2\n"
       << "stage\t" << stage << "\n"
       << "config_hash\t" << config_hash << "\n";
  for (const std::filesystem::path& required_file : required_files) {
    body << "required_file\t"
         << NormalizeRequiredFile(required_file).generic_string() << "\n";
  }
  for (const auto& [key, value] : values) {
    body << "value\t" << key << "\t" << value << "\n";
  }
  body << "artifact_fingerprint\t"
       << CacheArtifactFingerprint(workspace_path, required_files) << "\n";
  const auto text = body.str();
  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream file(temporary, std::ios::binary);
  file << text << "complete\t" << CacheBodyChecksum(text) << "\n";
  file.close();
  if (!file) throw std::runtime_error("Failed to write workspace cache: " + path.string());
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path);
    std::filesystem::rename(temporary, path);
  }
}

WorkspaceCacheMetadata ReadWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage) {
  const std::filesystem::path path =
      WorkspaceCacheMetadataPath(workspace_path, stage);
  WorkspaceCacheMetadata metadata;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return metadata;
  }

  std::string line, body;
  bool schema = false, complete = false;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string kind;
    std::getline(stream, kind, '\t');
    if (kind == "complete") {
      std::string checksum;
      std::getline(stream, checksum);
      complete = checksum == CacheBodyChecksum(body) && file.peek() == std::char_traits<char>::eof();
      break;
    }
    body += line + '\n';
    if (kind == "schema_version") {
      std::string version;
      std::getline(stream, version);
      if (schema || version != "2") return {};
      schema = true;
    } else if (kind == "stage") {
      if (!metadata.stage.empty()) return {};
      std::getline(stream, metadata.stage);
    } else if (kind == "config_hash") {
      if (!metadata.config_hash.empty()) return {};
      std::getline(stream, metadata.config_hash);
    } else if (kind == "required_file") {
      std::string required_file;
      std::getline(stream, required_file);
      metadata.required_files.emplace_back(required_file);
    } else if (kind == "artifact_fingerprint") {
      if (!metadata.required_files_fingerprint.empty()) return {};
      std::getline(stream, metadata.required_files_fingerprint);
    } else if (kind == "value") {
      std::string key;
      std::string value;
      std::getline(stream, key, '\t');
      std::getline(stream, value);
      if (key.empty() || !metadata.values.emplace(std::move(key), std::move(value)).second) return {};
    } else {
      return {};
    }
  }
  if (!schema || !complete || file.bad() || metadata.stage != stage ||
      metadata.config_hash.empty() || metadata.required_files_fingerprint.empty()) return {};
  return metadata;
}

bool WorkspaceCacheIsValid(const std::filesystem::path& workspace_path,
                           const std::string& stage,
                           const std::string& config_hash) {
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, stage);
  if (metadata.stage != stage || metadata.config_hash != config_hash) {
    return false;
  }
  if (const auto success = metadata.values.find("success");
      success != metadata.values.end() && success->second != "1") return false;
  return metadata.required_files_fingerprint ==
      CacheArtifactFingerprint(workspace_path, metadata.required_files);
}

void WriteWorkspaceCameras(const std::filesystem::path& workspace_path,
                           const std::vector<Camera>& cameras) {
  std::filesystem::create_directories(workspace_path);
  std::ofstream file(workspace_path / "cameras.txt");
  file << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] PRIOR_FOCAL=<0|1>\n";
  for (const Camera& camera : cameras) {
    file << camera.id << " " << CameraModelIdToName(camera.model_id) << " "
         << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      file << " " << std::setprecision(17) << param;
    }
    file << " PRIOR_FOCAL=" << (camera.has_prior_focal_length ? 1 : 0)
         << "\n";
  }
}

std::vector<Camera> ReadWorkspaceCameras(
    const std::filesystem::path& workspace_path) {
  std::vector<Camera> cameras;
  for (const std::string& line : DataLines(workspace_path / "cameras.txt")) {
    std::istringstream stream(line);
    Camera camera;
    std::string model_name;
    stream >> camera.id >> model_name >> camera.width >> camera.height;
    camera.model_id = CameraModelNameToId(model_name);
    const std::size_t num_params =
        GetCameraModelSpec(camera.model_id).param_names.size();
    for (std::size_t idx = 0; idx < num_params; ++idx) {
      double param = 0.0;
      if (!(stream >> param)) {
        throw std::runtime_error("Invalid workspace camera entry: " + line);
      }
      camera.params.push_back(param);
    }
    std::string metadata;
    if (stream >> metadata) {
      if (metadata == "PRIOR_FOCAL=1") {
        camera.has_prior_focal_length = true;
      } else if (metadata != "PRIOR_FOCAL=0") {
        throw std::runtime_error("Invalid workspace camera metadata: " + line);
      }
    }
    std::string trailing;
    if (stream >> trailing || !camera.VerifyParams()) {
      throw std::runtime_error("Invalid workspace camera entry: " + line);
    }
    cameras.push_back(camera);
  }
  return cameras;
}

void WriteWorkspaceImages(const std::filesystem::path& workspace_path,
                          const std::vector<Image>& images) {
  std::filesystem::create_directories(workspace_path);
  std::ofstream file(workspace_path / "images.tsv");
  file << "# image_id\tcamera_id\tname\tfile_path\tprior_x\tprior_y\tprior_z\n";
  for (const Image& image : images) {
    file << image.id << "\t" << image.camera_id << "\t" << image.name << "\t"
         << image.file_path;
    if (image.prior_position.has_value()) {
      file << "\t" << std::setprecision(17) << image.prior_position->x()
           << "\t" << image.prior_position->y() << "\t"
           << image.prior_position->z();
    }
    file << "\n";
  }
}

std::vector<Image> ReadWorkspaceImages(
    const std::filesystem::path& workspace_path) {
  std::vector<Image> images;
  for (const std::string& line : DataLines(workspace_path / "images.tsv")) {
    std::istringstream stream(line);
    Image image;
    std::string image_id_text;
    std::string camera_id_text;
    std::getline(stream, image_id_text, '\t');
    std::getline(stream, camera_id_text, '\t');
    image.id = static_cast<ImageId>(std::stoul(image_id_text));
    image.camera_id = static_cast<CameraId>(std::stoul(camera_id_text));
    std::getline(stream, image.name, '\t');
    std::getline(stream, image.file_path, '\t');
    if (!stream.eof()) {
      std::string x_text;
      std::string y_text;
      std::string z_text;
      std::getline(stream, x_text, '\t');
      std::getline(stream, y_text, '\t');
      std::getline(stream, z_text);
      if (x_text.empty() || y_text.empty() || z_text.empty()) {
        throw std::runtime_error("Invalid workspace image prior entry: " +
                                 line);
      }
      image.prior_position = Eigen::Vector3d(std::stod(x_text),
                                             std::stod(y_text),
                                             std::stod(z_text));
    }
    if (image.id == kInvalidImageId || image.camera_id == kInvalidCameraId ||
        image.name.empty()) {
      throw std::runtime_error("Invalid workspace image entry: " + line);
    }
    images.push_back(image);
  }
  return images;
}

void WriteWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                              const ImageId image_id,
                              const FeatureSet& features) {
  if (features.descriptor_source) {
    throw std::invalid_argument("Materialize deferred descriptors before saving features");
  }
  const auto source_columns = features.source_keypoint_columns;
  if ((source_columns != 0 && source_columns != 2 && source_columns != 4 && source_columns != 6) ||
      ((source_columns == 0 || source_columns == 2) && features.source_keypoint_shapes.size() != 0) ||
      ((source_columns == 4 || source_columns == 6) &&
       (features.source_keypoint_shapes.rows() != static_cast<Eigen::Index>(features.keypoints.size()) ||
        features.source_keypoint_shapes.cols() != source_columns - 2)) ||
      !features.source_keypoint_shapes.allFinite()) {
    throw std::invalid_argument("Invalid original keypoint shapes");
  }
  if (features.descriptors.size() != 0 &&
      features.descriptors.rows() !=
          static_cast<Eigen::Index>(features.keypoints.size())) {
    throw std::invalid_argument(
        "Feature keypoint and descriptor row counts do not match");
  }
  std::filesystem::create_directories(WorkspaceFeaturesPath(workspace_path));
  std::ofstream file(WorkspaceFeaturesPath(workspace_path) /
                         (std::to_string(image_id) + ".bin"),
                     std::ios::binary);
  const std::uint32_t extractor_type =
      static_cast<std::uint32_t>(features.extractor_type);
  const std::uint32_t scalar_type =
      static_cast<std::uint32_t>(features.descriptor_scalar_type);
  const std::uint32_t descriptor_metric =
      static_cast<std::uint32_t>(features.descriptor_metric);
  const std::uint32_t descriptor_normalization =
      static_cast<std::uint32_t>(features.descriptor_normalization);
  const std::uint64_t num_keypoints = features.keypoints.size();
  const std::uint64_t rows = static_cast<std::uint64_t>(
      features.descriptors.rows());
  const std::uint64_t cols = static_cast<std::uint64_t>(
      features.descriptors.cols());
  const std::uint64_t payload_bytes =
      rows * cols * static_cast<std::uint64_t>(sizeof(float));
  const std::uint64_t payload_checksum = FeaturePayloadChecksum(features, true);
  WriteBinary(&file, kFeatureFileMagic);
  WriteBinary(&file, kFeatureFileVersion);
  WriteBinary(&file, extractor_type);
  WriteBinary(&file, scalar_type);
  WriteBinary(&file, descriptor_metric);
  WriteBinary(&file, descriptor_normalization);
  WriteBinary(&file, num_keypoints);
  WriteBinary(&file, rows);
  WriteBinary(&file, cols);
  WriteBinary(&file, payload_bytes);
  WriteBinary(&file, payload_checksum);
  WriteBinary(&file, source_columns);
  WriteBinary(&file, std::uint32_t{0});  // Reserved, must be zero.
  for (const FeatureKeypoint& keypoint : features.keypoints) {
    WriteBinary(&file, keypoint.x);
    WriteBinary(&file, keypoint.y);
    WriteBinary(&file, keypoint.scale);
    WriteBinary(&file, keypoint.orientation);
    WriteBinary(&file, keypoint.response);
  }
  if (features.source_keypoint_shapes.size() > 0) {
    file.write(reinterpret_cast<const char*>(features.source_keypoint_shapes.data()),
               static_cast<std::streamsize>(features.source_keypoint_shapes.size() * sizeof(float)));
  }
  if (features.descriptors.size() > 0) {
    file.write(reinterpret_cast<const char*>(features.descriptors.data()),
               static_cast<std::streamsize>(features.descriptors.size() *
                                            sizeof(float)));
  }
}

FeatureSet ReadWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                                   const ImageId image_id,
                                   const WorkspaceFeatureLoading loading) {
  const auto path = WorkspaceFeaturesPath(workspace_path) /
                    (std::to_string(image_id) + ".bin");
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Missing feature file for image " +
                             std::to_string(image_id));
  }
  const auto file_bytes = std::filesystem::file_size(path);
  const auto write_time = std::filesystem::last_write_time(path);
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t num_keypoints = 0;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  ReadBinary(&file, &magic);
  ReadBinary(&file, &version);
  ReadBinary(&file, &num_keypoints);
  ReadBinary(&file, &rows);
  ReadBinary(&file, &cols);
  if (magic != kFeatureFileMagic || (version != 1 && version != 2 && version != 3)) {
    throw std::runtime_error("Unsupported feature file format");
  }

  FeatureSet features;
  std::uint64_t payload_bytes = 0;
  std::uint64_t expected_checksum = 0;
  if (version >= 2) {
    std::uint32_t extractor_type = 0;
    std::uint32_t scalar_type = 0;
    std::uint32_t descriptor_metric = 0;
    std::uint32_t descriptor_normalization = 0;
    // Version 2 metadata precedes the dimensions. Re-read the header from the
    // first field after version, since version 1 placed dimensions here.
    file.seekg(static_cast<std::streamoff>(2 * sizeof(std::uint32_t)),
               std::ios::beg);
    ReadBinary(&file, &extractor_type);
    ReadBinary(&file, &scalar_type);
    ReadBinary(&file, &descriptor_metric);
    ReadBinary(&file, &descriptor_normalization);
    ReadBinary(&file, &num_keypoints);
    ReadBinary(&file, &rows);
    ReadBinary(&file, &cols);
    ReadBinary(&file, &payload_bytes);
    ReadBinary(&file, &expected_checksum);
    features.extractor_type = ParseFeatureExtractorType(extractor_type);
    features.descriptor_scalar_type =
        ParseDescriptorScalarType(scalar_type);
    features.descriptor_metric = ParseDescriptorMetric(descriptor_metric);
    features.descriptor_normalization =
        ParseDescriptorNormalization(descriptor_normalization);
  }
  if (version >= 3) {
    std::uint32_t reserved = 0;
    ReadBinary(&file, &features.source_keypoint_columns);
    ReadBinary(&file, &reserved);
    const auto columns = features.source_keypoint_columns;
    if (reserved != 0 || (columns != 0 && columns != 2 && columns != 4 && columns != 6)) {
      throw std::runtime_error("Invalid original keypoint shape metadata");
    }
  }
  if (num_keypoints > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      (rows != 0 && rows != num_keypoints) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      cols > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      (cols != 0 && rows > std::numeric_limits<std::uint64_t>::max() / cols) ||
      rows * cols > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    throw std::runtime_error("Invalid feature matrix dimensions");
  }
  const std::uint64_t expected_payload_bytes = rows * cols * sizeof(float);
  if (expected_payload_bytes > kMaxFeatureFilePayloadBytes) {
    throw std::runtime_error("Feature descriptor payload is unreasonably large");
  }
  if (version >= 2 && payload_bytes != expected_payload_bytes) {
    throw std::runtime_error("Invalid descriptor payload size");
  }
  const auto header_bytes = static_cast<std::uint64_t>(file.tellg());
  const auto source_columns = features.source_keypoint_columns;
  const std::uint64_t shape_columns = source_columns >= 4 ? source_columns - 2 : 0;
  const std::uint64_t kKeypointBytes = (5 + shape_columns) * sizeof(float);
  if (header_bytes > file_bytes ||
      num_keypoints > (file_bytes - header_bytes) / kKeypointBytes) {
    throw std::runtime_error("Truncated feature keypoints");
  }
  const auto descriptor_offset = header_bytes + num_keypoints * kKeypointBytes;
  if (file_bytes - descriptor_offset != expected_payload_bytes) {
    throw std::runtime_error("Feature file size does not match its payload");
  }
  features.keypoints.resize(static_cast<std::size_t>(num_keypoints));
  for (FeatureKeypoint& keypoint : features.keypoints) {
    ReadBinary(&file, &keypoint.x);
    ReadBinary(&file, &keypoint.y);
    ReadBinary(&file, &keypoint.scale);
    ReadBinary(&file, &keypoint.orientation);
    ReadBinary(&file, &keypoint.response);
  }
  if (shape_columns > 0) {
    features.source_keypoint_shapes.resize(static_cast<int>(num_keypoints), static_cast<int>(shape_columns));
    file.read(reinterpret_cast<char*>(features.source_keypoint_shapes.data()),
              static_cast<std::streamsize>(num_keypoints * shape_columns * sizeof(float)));
    if (!file || !features.source_keypoint_shapes.allFinite()) {
      throw std::runtime_error("Invalid original keypoint shape payload");
    }
  }
  if (loading == WorkspaceFeatureLoading::kDeferredDescriptors) {
    std::uint64_t checksum = FeaturePayloadChecksum(features, version >= 3);
    std::array<char, 64 * 1024> buffer;
    for (std::uint64_t remaining = expected_payload_bytes; remaining > 0;) {
      const auto count = static_cast<std::size_t>(
          std::min<std::uint64_t>(remaining, buffer.size()));
      file.read(buffer.data(), static_cast<std::streamsize>(count));
      if (!file) throw std::runtime_error("Truncated descriptor matrix");
      if (version >= 2) HashBytes(&checksum, buffer.data(), count);
      remaining -= count;
    }
    if (version >= 2 && checksum != expected_checksum) {
      throw std::runtime_error("Feature payload checksum mismatch");
    }
    if (std::filesystem::file_size(path) != file_bytes ||
        std::filesystem::last_write_time(path) != write_time) {
      throw std::runtime_error("Feature file changed during validation");
    }
    features.descriptor_source = std::make_shared<FileDescriptorSource>(
        path, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols),
        static_cast<std::streamoff>(descriptor_offset), file_bytes, write_time);
    return features;
  }
  features.descriptors.resize(static_cast<int>(rows), static_cast<int>(cols));
  if (features.descriptors.size() > 0) {
    file.read(reinterpret_cast<char*>(features.descriptors.data()),
              static_cast<std::streamsize>(features.descriptors.size() *
                                           sizeof(float)));
    if (!file) {
      throw std::runtime_error("Truncated descriptor matrix");
    }
  }
  if (version >= 2 && FeaturePayloadChecksum(features, version >= 3) != expected_checksum) {
    throw std::runtime_error("Feature payload checksum mismatch");
  }
  if (file.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Feature file has trailing data");
  }
  return features;
}

void ReadWorkspaceFeatureSets(const std::filesystem::path& workspace_path,
                              MatchGraph* graph,
                              const WorkspaceFeatureLoading loading) {
  for (const Image& image : ReadWorkspaceImages(workspace_path)) {
    graph->features.insert_or_assign(
        image.id, ReadWorkspaceFeatureSet(workspace_path, image.id, loading));
  }
}

void WriteWorkspaceMatches(const std::filesystem::path& workspace_path,
                           const MatchGraph& graph) {
  std::filesystem::create_directories(WorkspaceMatchesPath(workspace_path));
  std::vector<char> buffer(1024 * 1024);
  std::ofstream file;
  file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
  file.open(WorkspaceMatchesPath(workspace_path) / "raw_matches.tsv");
  file << "# image_id1\timage_id2\tidx1\tidx2\n";
  for (const auto& [pair, matches] : graph.matches) {
    if (matches.empty()) {
      file << pair.image_id1 << '\t' << pair.image_id2 << "\t-1\t-1\n";
    }
    for (const FeatureMatch& match : matches) {
      file << pair.image_id1 << "\t" << pair.image_id2 << "\t" << match.idx1
           << "\t" << match.idx2 << "\n";
    }
  }
}

void ReadWorkspaceMatches(const std::filesystem::path& workspace_path,
                          MatchGraph* graph) {
  std::map<ImagePair, std::vector<FeatureMatch>> grouped;
  const std::filesystem::path path =
      WorkspaceMatchesPath(workspace_path) / "raw_matches.tsv";
  if (!std::filesystem::exists(path)) {
    return;
  }
  std::vector<char> buffer(1024 * 1024);
  std::ifstream file;
  file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
  file.open(path);
  if (!file) throw std::runtime_error("Failed to read raw matches: " + path.string());
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') continue;
    std::istringstream stream(line);
    ImageId image_id1 = kInvalidImageId;
    ImageId image_id2 = kInvalidImageId;
    FeatureMatch match;
    stream >> image_id1 >> image_id2 >> match.idx1 >> match.idx2;
    if (!stream) {
      throw std::runtime_error("Invalid raw match entry: " + line);
    }
    auto& matches = grouped[ImagePair::Ordered(image_id1, image_id2)];
    if (match.idx1 == kInvalidPoint2DIdx && match.idx2 == kInvalidPoint2DIdx) continue;
    if (match.idx1 == kInvalidPoint2DIdx || match.idx2 == kInvalidPoint2DIdx) {
      throw std::runtime_error("Invalid raw match sentinel: " + line);
    }
    if (image_id1 > image_id2) std::swap(match.idx1, match.idx2);
    matches.push_back(match);
  }
  for (auto& [pair, matches] : grouped) {
    if (pair.image_id1 == pair.image_id2) {
      throw std::invalid_argument("Cannot store matches for the same image");
    }
    graph->matches[pair] = std::move(matches);
  }
}

void WriteWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path, const MatchGraph& graph) {
  std::filesystem::create_directories(WorkspaceMatchesPath(workspace_path));
  std::vector<char> buffer(1024 * 1024);
  std::ofstream file;
  file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
  file.open(WorkspaceMatchesPath(workspace_path) / "verified_matches.tsv");
  if (!file) throw std::runtime_error("Failed to write verified geometries");
  file << "# COLITE_VERIFIED_GEOMETRIES 2\n"
          "# Columns: pair ID1 ID2 CONFIG FLAGS MEDIAN_ANGLE NUM_INLIERS F[9] E[9] H[9] QW QX QY QZ TX TY TZ\n"
          "# FLAGS: F=1 E=2 H=4 POSE=8; matrices are row-major.\n";
  file << std::setprecision(17);
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (pair.image_id1 >= pair.image_id2 || pair.image_id2 == kInvalidImageId) {
      throw std::runtime_error("Verified geometry pair must be canonical");
    }
    ValidateVerifiedGeometry(geometry);
    const int flags = (geometry.F ? 1 : 0) | (geometry.E ? 2 : 0) |
                      (geometry.H ? 4 : 0) | (geometry.cam2_from_cam1 ? 8 : 0);
    file << "# pair " << pair.image_id1 << ' ' << pair.image_id2 << ' '
         << GeometryTypeToInt(geometry.type) << ' ' << flags << ' '
         << geometry.median_triangulation_angle << ' ' << geometry.inlier_matches.size();
    for (const auto* matrix : {&geometry.F, &geometry.E, &geometry.H}) {
      for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
        file << ' ' << (*matrix ? (**matrix)(row, col) : 0.0);
      }
    }
    const Pose3 pose = geometry.cam2_from_cam1.value_or(Pose3{});
    file << ' ' << pose.q_cw.w() << ' ' << pose.q_cw.x() << ' ' << pose.q_cw.y()
         << ' ' << pose.q_cw.z() << ' ' << pose.t_cw.x() << ' ' << pose.t_cw.y()
         << ' ' << pose.t_cw.z() << '\n';
    const auto prefix = VerifiedPrefix(pair, geometry);
    if (geometry.inlier_matches.empty()) file << prefix << "-1\t-1\n";
    for (const auto& match : geometry.inlier_matches) {
      if (match.idx1 == kInvalidPoint2DIdx || match.idx2 == kInvalidPoint2DIdx) {
        throw std::runtime_error("Invalid verified match index");
      }
      file << prefix << match.idx1 << '\t' << match.idx2 << '\n';
    }
  }
  if (!file) throw std::runtime_error("Failed to write verified geometries");
}

void ReadWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path, MatchGraph* graph) {
  const auto path = WorkspaceMatchesPath(workspace_path) / "verified_matches.tsv";
  if (!std::filesystem::exists(path)) return;
  std::vector<char> buffer(1024 * 1024);
  std::ifstream file;
  file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
  file.open(path);
  if (!file) throw std::runtime_error("Failed to read verified matches: " + path.string());
  std::map<ImagePair, TwoViewGeometry> grouped;
  std::map<ImagePair, std::size_t> expected;
  std::set<ImagePair> empty_pairs;
  bool modern = false, data_seen = false;
  std::string line, previous_prefix;
  TwoViewGeometry* previous_geometry = nullptr;
  ImagePair previous_pair;
  bool previous_reversed = false;
  auto fail = [] { throw std::runtime_error("Invalid or inconsistent verified geometry record"); };
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line.rfind("# COLITE_VERIFIED_GEOMETRIES", 0) == 0) {
      if (line != "# COLITE_VERIFIED_GEOMETRIES 2" || modern || data_seen) fail();
      modern = true;
      continue;
    }
    if (line.rfind("# pair ", 0) == 0) {
      if (!modern) fail();
      std::istringstream stream(line.substr(7));
      ImagePair pair;
      int type = 0, flags = 0;
      std::int64_t count = -1;
      TwoViewGeometry geometry;
      if (!(stream >> pair.image_id1 >> pair.image_id2 >> type >> flags
                   >> geometry.median_triangulation_angle >> count) ||
          pair.image_id1 >= pair.image_id2 || pair.image_id2 == kInvalidImageId ||
          flags < 0 || flags > 15 || count < 0) fail();
      geometry.type = GeometryTypeFromInt(type);
      int bit = 1;
      for (auto* matrix : {&geometry.F, &geometry.E, &geometry.H}) {
        Eigen::Matrix3d values;
        for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
          if (!(stream >> values(row, col))) fail();
        }
        if (!values.allFinite() || (!(flags & bit) && values.norm() != 0.0)) fail();
        if (flags & bit) *matrix = values;
        bit <<= 1;
      }
      Pose3 pose;
      if (!(stream >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >> pose.q_cw.z()
                   >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z())) fail();
      std::string extra;
      if (stream >> extra) fail();
      if (flags & 8) geometry.cam2_from_cam1 = pose;
      else if (!pose.q_cw.coeffs().isApprox(Eigen::Quaterniond::Identity().coeffs(), 0.0) ||
               pose.t_cw.norm() != 0.0) fail();
      ValidateVerifiedGeometry(geometry);
      if (!grouped.empty() && !(grouped.rbegin()->first < pair)) fail();
      if (!grouped.emplace(pair, std::move(geometry)).second) fail();
      expected.emplace(pair, static_cast<std::size_t>(count));
      data_seen = true;
      previous_geometry = nullptr;
      continue;
    }
    if (line.front() == '#') continue;
    data_seen = true;
    std::int64_t idx1 = -2, idx2 = -2;
    if (previous_geometry && !previous_prefix.empty() &&
        line.compare(0, previous_prefix.size(), previous_prefix) == 0) {
      std::istringstream stream(line.substr(previous_prefix.size()));
      std::string extra;
      if (!(stream >> idx1 >> idx2) || (stream >> extra)) fail();
    } else {
      std::istringstream stream(line);
      ImageId id1 = kInvalidImageId, id2 = kInvalidImageId;
      int type = -1;
      TwoViewGeometry geometry;
      Pose3 pose;
      std::string extra;
      if (!(stream >> id1 >> id2 >> type >> geometry.median_triangulation_angle
                   >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >> pose.q_cw.z()
                   >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z() >> idx1 >> idx2) ||
          (stream >> extra) || id1 == id2 || id1 == kInvalidImageId || id2 == kInvalidImageId) fail();
      geometry.type = GeometryTypeFromInt(!modern && type == 6 ? 8 : type);
      geometry.cam2_from_cam1 = pose;
      ValidateVerifiedGeometry(geometry);
      const auto pair = ImagePair::Ordered(id1, id2);
      auto it = grouped.find(pair);
      if (modern) {
        if (id1 > id2 || it == grouped.end() ||
            geometry.type != it->second.type ||
            geometry.median_triangulation_angle != it->second.median_triangulation_angle) fail();
        const auto stored_pose = it->second.cam2_from_cam1.value_or(Pose3{});
        if (!stored_pose.q_cw.coeffs().isApprox(pose.q_cw.coeffs(), 0.0) ||
            !stored_pose.t_cw.isApprox(pose.t_cw, 0.0)) fail();
      } else {
        // Legacy rows had no presence flags: retain only demonstrable poses.
        if (pose.t_cw.norm() <= 1e-12 && geometry.type != TwoViewGeometry::Type::kPanoramic) {
          geometry.cam2_from_cam1.reset();
        }
        if (id1 > id2 && geometry.cam2_from_cam1) {
          geometry.cam2_from_cam1 = geometry.cam2_from_cam1->Inverse();
        }
        if (it == grouped.end()) it = grouped.emplace(pair, geometry).first;
        else if (VerifiedPrefix(pair, geometry) != VerifiedPrefix(pair, it->second)) fail();
      }
      previous_pair = pair;
      previous_reversed = id1 > id2;
      previous_geometry = &it->second;
      const auto last = line.rfind('\t');
      const auto tab = last == std::string::npos || last == 0 ? std::string::npos : line.rfind('\t', last - 1);
      previous_prefix = tab == std::string::npos ? "" : line.substr(0, tab + 1);
    }
    if (idx1 == -1 && idx2 == -1) {
      if (!previous_geometry->inlier_matches.empty() || !empty_pairs.insert(previous_pair).second) fail();
    } else {
      if (idx1 < 0 || idx2 < 0 || idx1 >= kInvalidPoint2DIdx || idx2 >= kInvalidPoint2DIdx ||
          empty_pairs.count(previous_pair)) fail();
      if (previous_reversed) std::swap(idx1, idx2);
      previous_geometry->inlier_matches.push_back({static_cast<Point2DIdx>(idx1), static_cast<Point2DIdx>(idx2)});
    }
  }
  if (!file.eof()) throw std::runtime_error("Failed while reading verified geometries");
  for (const auto& [pair, geometry] : grouped) {
    if (modern && (geometry.inlier_matches.size() != expected.at(pair) ||
                   (geometry.inlier_matches.empty() && !empty_pairs.count(pair)))) fail();
  }
  // Validate every pair before changing the caller's graph.
  for (auto& [pair, geometry] : grouped) {
    graph->verified_geometries[pair] = std::move(geometry);
  }
}

void WriteWorkspaceSparseReconstruction(
    const std::filesystem::path& workspace_path,
    const Reconstruction& reconstruction,
    const ColmapModelFormat format) {
  WriteColmapReconstruction(WorkspaceSparsePath(workspace_path),
                            reconstruction, format);
}

void WriteSparseReconstruction(
    const std::filesystem::path& sparse_path,
    const Reconstruction& reconstruction) {
  std::filesystem::create_directories(sparse_path);

  std::ofstream cameras_file(sparse_path / "cameras.txt");
  cameras_file.exceptions(std::ios::failbit | std::ios::badbit);
  cameras_file.imbue(std::locale::classic());
  cameras_file << "# Camera list with one line of data per camera:\n"
               << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n";
  for (const CameraId camera_id : SortedIds<CameraId>(reconstruction.Cameras())) {
    const Camera& camera = reconstruction.GetCamera(camera_id);
    cameras_file << camera.id << " " << CameraModelIdToName(camera.model_id)
                 << " " << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      cameras_file << " " << std::setprecision(17) << param;
    }
    cameras_file << "\n";
  }

  std::ofstream images_file(sparse_path / "images.txt");
  images_file.exceptions(std::ios::failbit | std::ios::badbit);
  images_file.imbue(std::locale::classic());
  images_file << "# Image list with two lines of data per image:\n"
              << "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
              << "# POINTS2D[] as X Y POINT3D_ID\n";
  for (const ImageId image_id : reconstruction.RegisteredImageIds()) {
    const Image& image = reconstruction.GetImage(image_id);
    if (!image.registered || !image.cam_from_world.has_value()) {
      throw std::runtime_error("Registered image has no pose while writing COLMAP text model");
    }
    const Pose3& pose = *image.cam_from_world;
    images_file << std::setprecision(17) << image_id << " "
                << pose.q_cw.w() << " " << pose.q_cw.x() << " "
                << pose.q_cw.y() << " " << pose.q_cw.z() << " "
                << pose.t_cw.x() << " " << pose.t_cw.y() << " "
                << pose.t_cw.z() << " " << image.camera_id << " "
                << image.name << "\n";
    for (std::size_t i = 0; i < image.points2d.size(); ++i) {
      if (i > 0) {
        images_file << " ";
      }
      images_file << std::setprecision(17) << image.points2d[i].xy.x()
                  << " " << image.points2d[i].xy.y() << " "
                  << Point3DIdForText(image.points2d[i].point3d_id);
    }
    images_file << "\n";
  }

  std::ofstream points_file(sparse_path / "points3D.txt");
  points_file.exceptions(std::ios::failbit | std::ios::badbit);
  points_file.imbue(std::locale::classic());
  points_file << "# 3D point list with one line of data per point:\n"
              << "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n";
  for (const Point3DId point3d_id : SortedIds<Point3DId>(reconstruction.Points3D())) {
    const Point3D& point3d = reconstruction.GetPoint3D(point3d_id);
    points_file << point3d_id << " " << std::setprecision(17)
                << point3d.xyz.x() << " " << point3d.xyz.y() << " "
                << point3d.xyz.z() << " "
                << static_cast<int>(point3d.color.x()) << " "
                << static_cast<int>(point3d.color.y()) << " "
                << static_cast<int>(point3d.color.z()) << " "
                << point3d.reprojection_error;
    for (const TrackElement& element : point3d.track.elements) {
      points_file << " " << element.image_id << " " << element.point2d_idx;
    }
    points_file << "\n";
  }

  std::ofstream ply_file(sparse_path / "points3D.ply");
  ply_file.exceptions(std::ios::failbit | std::ios::badbit);
  ply_file.imbue(std::locale::classic());
  ply_file << "ply\nformat ascii 1.0\nelement vertex "
           << reconstruction.Points3D().size()
           << "\nproperty double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n";
  for (const Point3DId point3d_id : SortedIds<Point3DId>(reconstruction.Points3D())) {
    const Point3D& point3d = reconstruction.GetPoint3D(point3d_id);
    (void)point3d_id;
    ply_file << std::setprecision(17) << point3d.xyz.x() << " "
             << point3d.xyz.y() << " " << point3d.xyz.z() << " "
             << static_cast<int>(point3d.color.x()) << " "
             << static_cast<int>(point3d.color.y()) << " "
             << static_cast<int>(point3d.color.z()) << "\n";
  }
  cameras_file.close();
  images_file.close();
  points_file.close();
  ply_file.close();
}

void WriteColmapBinaryReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction) {
  std::filesystem::create_directories(model_path);
  std::ofstream cameras_file(model_path / "cameras.bin", std::ios::binary);
  std::ofstream images_file(model_path / "images.bin", std::ios::binary);
  std::ofstream points_file(model_path / "points3D.bin", std::ios::binary);
  if (!cameras_file || !images_file || !points_file) {
    throw std::runtime_error("Failed to create COLMAP binary model: " +
                             model_path.string());
  }

  const std::vector<CameraId> camera_ids =
      SortedIds<CameraId>(reconstruction.Cameras());
  WriteColmapValue(&cameras_file, static_cast<std::uint64_t>(camera_ids.size()));
  for (const CameraId camera_id : camera_ids) {
    const Camera& camera = reconstruction.GetCamera(camera_id);
    WriteColmapValue(&cameras_file, static_cast<std::uint32_t>(camera.id));
    WriteColmapValue(&cameras_file, static_cast<std::int32_t>(camera.model_id));
    WriteColmapValue(&cameras_file, static_cast<std::uint64_t>(camera.width));
    WriteColmapValue(&cameras_file, static_cast<std::uint64_t>(camera.height));
    for (const double parameter : camera.params) {
      WriteColmapValue(&cameras_file, parameter);
    }
  }

  std::vector<ImageId> registered_image_ids(
      reconstruction.RegisteredImageIds().begin(),
      reconstruction.RegisteredImageIds().end());
  WriteColmapValue(&images_file,
              static_cast<std::uint64_t>(registered_image_ids.size()));
  for (const ImageId image_id : registered_image_ids) {
    const Image& image = reconstruction.GetImage(image_id);
    if (!image.cam_from_world.has_value()) {
      throw std::runtime_error(
          "Registered image has no pose while writing COLMAP binary model");
    }
    const Pose3& pose = *image.cam_from_world;
    WriteColmapValue(&images_file, static_cast<std::uint32_t>(image.id));
    WriteColmapValue(&images_file, pose.q_cw.w());
    WriteColmapValue(&images_file, pose.q_cw.x());
    WriteColmapValue(&images_file, pose.q_cw.y());
    WriteColmapValue(&images_file, pose.q_cw.z());
    WriteColmapValue(&images_file, pose.t_cw.x());
    WriteColmapValue(&images_file, pose.t_cw.y());
    WriteColmapValue(&images_file, pose.t_cw.z());
    WriteColmapValue(&images_file, static_cast<std::uint32_t>(image.camera_id));
    images_file.write(image.name.data(),
                      static_cast<std::streamsize>(image.name.size()));
    images_file.put('\0');
    if (!images_file) {
      throw std::runtime_error("Failed to write COLMAP binary image name");
    }
    WriteColmapValue(&images_file,
                static_cast<std::uint64_t>(image.points2d.size()));
    for (const Point2D& point2d : image.points2d) {
      WriteColmapValue(&images_file, point2d.xy.x());
      WriteColmapValue(&images_file, point2d.xy.y());
      WriteColmapValue(&images_file,
                  static_cast<std::uint64_t>(point2d.point3d_id));
    }
  }

  const std::vector<Point3DId> point_ids =
      SortedIds<Point3DId>(reconstruction.Points3D());
  WriteColmapValue(&points_file, static_cast<std::uint64_t>(point_ids.size()));
  for (const Point3DId point_id : point_ids) {
    const Point3D& point = reconstruction.GetPoint3D(point_id);
    WriteColmapValue(&points_file, static_cast<std::uint64_t>(point.id));
    WriteColmapValue(&points_file, point.xyz.x());
    WriteColmapValue(&points_file, point.xyz.y());
    WriteColmapValue(&points_file, point.xyz.z());
    WriteColmapValue(&points_file, point.color.x());
    WriteColmapValue(&points_file, point.color.y());
    WriteColmapValue(&points_file, point.color.z());
    WriteColmapValue(&points_file, point.reprojection_error);
    WriteColmapValue(&points_file,
                static_cast<std::uint64_t>(point.track.elements.size()));
    for (const TrackElement& element : point.track.elements) {
      WriteColmapValue(&points_file, static_cast<std::uint32_t>(element.image_id));
      WriteColmapValue(&points_file,
                  static_cast<std::uint32_t>(element.point2d_idx));
    }
  }
  cameras_file.close();
  images_file.close();
  points_file.close();
  if (!cameras_file || !images_file || !points_file) {
    throw std::runtime_error("Failed to finish COLMAP binary model");
  }
}

void WriteColmapReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction,
    const ColmapModelFormat format) {
  switch (format) {
    case ColmapModelFormat::kText:
      WriteSparseReconstruction(model_path, reconstruction);
      for (const char* name : {"cameras.bin", "images.bin", "points3D.bin"}) {
        std::filesystem::remove(model_path / name);
      }
      break;
    case ColmapModelFormat::kBinary:
      WriteColmapBinaryReconstruction(model_path, reconstruction);
      for (const char* name : {"cameras.txt", "images.txt", "points3D.txt"}) {
        std::filesystem::remove(model_path / name);
      }
      break;
    case ColmapModelFormat::kTextAndBinary:
      WriteSparseReconstruction(model_path, reconstruction);
      WriteColmapBinaryReconstruction(model_path, reconstruction);
      break;
    case ColmapModelFormat::kAuto:
      throw std::invalid_argument(
          "COLMAP auto format is valid for reading but not writing");
    default:
      throw std::invalid_argument("Unknown COLMAP model format");
  }
  // This writer emits the legacy three-file model. COLMAP 4 can synthesize
  // trivial rigs/frames; stale metadata from a previous model would override
  // the exported image poses and must not accompany these files.
  for (const char* name : {"rigs.txt", "frames.txt", "rigs.bin", "frames.bin"}) {
    std::filesystem::remove(model_path / name);
  }
}

namespace {
// The 0.1 reconstruction model represents independent cameras. Accept the
// trivial rigs emitted by COLMAP 4, use authoritative frame poses/order, and
// reject coupled sensors instead of silently discarding their constraints.
void ApplyTrivialColmapFrames(const std::filesystem::path& root, bool binary,
                              Reconstruction* reconstruction,
                              std::vector<ImageId>* image_order) {
  const auto rigs_path = root / (binary ? "rigs.bin" : "rigs.txt");
  const auto frames_path = root / (binary ? "frames.bin" : "frames.txt");
  const bool has_rigs = std::filesystem::exists(rigs_path);
  const bool has_frames = std::filesystem::exists(frames_path);
  if (!has_rigs && !has_frames) return;
  const auto invalid = [] {
    throw std::runtime_error("Unsupported or invalid COLMAP rig/frame model: Colite 0.1 supports one reference camera per rig and one image per frame");
  };
  if (!has_rigs) invalid();
  std::map<std::uint32_t, CameraId> rigs;
  const auto add_rig = [&](std::uint32_t id, std::uint32_t count,
                            bool camera_type, CameraId camera) {
    if (count != 1 || !camera_type || !reconstruction->ExistsCamera(camera) ||
        !rigs.emplace(id, camera).second) invalid();
  };
  if (binary) {
    std::ifstream file(rigs_path, std::ios::binary);
    std::uint64_t count = 0;
    ReadColmapValue(&file, &count);
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint32_t id = 0, num_sensors = 0, camera = 0;
      std::int32_t type = -1;
      ReadColmapValue(&file, &id);
      ReadColmapValue(&file, &num_sensors);
      if (num_sensors != 1) invalid();
      ReadColmapValue(&file, &type);
      ReadColmapValue(&file, &camera);
      add_rig(id, num_sensors, type == 0, camera);
    }
    if (file.peek() != std::char_traits<char>::eof()) invalid();
  } else {
    for (const auto& line : DataLines(rigs_path)) {
      std::istringstream stream(line);
      std::uint32_t id = 0, count = 0, camera = 0;
      std::string type;
      if (!(stream >> id >> count >> type >> camera)) invalid();
      stream >> std::ws;
      if (!stream.eof()) invalid();
      add_rig(id, count, type == "CAMERA", camera);
    }
  }
  if (!has_frames) return;
  std::set<std::uint32_t> frame_ids;
  std::set<ImageId> seen;
  std::vector<ImageId> order;
  const auto add_frame = [&](std::uint32_t frame, std::uint32_t rig,
      Pose3 pose, std::uint32_t count, bool camera_type, CameraId camera,
      std::uint64_t image_id) {
    if (count != 1 || !camera_type || !rigs.count(rig) || rigs.at(rig) != camera ||
        image_id > std::numeric_limits<ImageId>::max() ||
        !reconstruction->ExistsImage(static_cast<ImageId>(image_id)) ||
        !frame_ids.insert(frame).second || !seen.insert(static_cast<ImageId>(image_id)).second ||
        !pose.q_cw.coeffs().allFinite() || !pose.t_cw.allFinite() ||
        std::abs(pose.q_cw.norm() - 1.0) > 1e-6) invalid();
    auto& image = reconstruction->GetImage(static_cast<ImageId>(image_id));
    if (image.camera_id != camera) invalid();
    image.cam_from_world = pose;
    order.push_back(image.id);
  };
  if (binary) {
    std::ifstream file(frames_path, std::ios::binary);
    std::uint64_t count = 0;
    ReadColmapValue(&file, &count);
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint32_t id = 0, rig = 0, num_data = 0, camera = 0;
      std::int32_t type = -1;
      std::uint64_t image = 0;
      Pose3 pose;
      ReadColmapValue(&file, &id); ReadColmapValue(&file, &rig);
      ReadColmapValue(&file, &pose.q_cw.w()); ReadColmapValue(&file, &pose.q_cw.x());
      ReadColmapValue(&file, &pose.q_cw.y()); ReadColmapValue(&file, &pose.q_cw.z());
      ReadColmapValue(&file, &pose.t_cw.x()); ReadColmapValue(&file, &pose.t_cw.y());
      ReadColmapValue(&file, &pose.t_cw.z()); ReadColmapValue(&file, &num_data);
      if (num_data != 1) invalid();
      ReadColmapValue(&file, &type); ReadColmapValue(&file, &camera); ReadColmapValue(&file, &image);
      add_frame(id, rig, pose, num_data, type == 0, camera, image);
    }
    if (file.peek() != std::char_traits<char>::eof()) invalid();
  } else {
    for (const auto& line : DataLines(frames_path)) {
      std::istringstream stream(line);
      stream.imbue(std::locale::classic());
      std::uint32_t id = 0, rig = 0, count = 0, camera = 0;
      std::uint64_t image = 0;
      std::string type;
      Pose3 pose;
      if (!(stream >> id >> rig >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >> pose.q_cw.z()
            >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z() >> count >> type >> camera >> image)) invalid();
      stream >> std::ws;
      if (!stream.eof()) invalid();
      add_frame(id, rig, pose, count, type == "CAMERA", camera, image);
    }
  }
  if (order.size() != reconstruction->RegisteredImageIds().size()) invalid();
  if (image_order) *image_order = std::move(order);
}
}  // namespace

Reconstruction ReadColmapTextReconstruction(
    const std::filesystem::path& model_path,
    std::vector<ImageId>* image_ids_in_model_order) {
  if (image_ids_in_model_order) image_ids_in_model_order->clear();
  Reconstruction reconstruction;
  for (const std::string& line : DataLines(model_path / "cameras.txt")) {
    std::istringstream stream(line);
    stream.imbue(std::locale::classic());
    Camera camera;
    std::string model_name;
    stream >> camera.id >> model_name >> camera.width >> camera.height;
    camera.model_id = CameraModelNameToId(model_name);
    double parameter = 0.0;
    while (stream >> parameter) {
      camera.params.push_back(parameter);
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid COLMAP camera entry: " + line);
    }
    reconstruction.AddCamera(std::move(camera));
  }

  for (const auto& [header_line, points_line] :
       ColmapImageEntries(model_path / "images.txt")) {
    std::istringstream header(header_line);
    header.imbue(std::locale::classic());
    Image image;
    Pose3 pose;
    header >> image.id >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >>
        pose.q_cw.z() >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z() >>
        image.camera_id;
    std::getline(header, image.name);
    const std::size_t name_start = image.name.find_first_not_of(" \t");
    if (name_start != std::string::npos) {
      image.name.erase(0, name_start);
    } else {
      image.name.clear();
    }
    if (!header || image.name.empty() || !reconstruction.ExistsCamera(image.camera_id)) {
      throw std::runtime_error("Invalid COLMAP image entry: " + header_line);
    }
    image.cam_from_world = pose;
    image.registered = true;
    if (image_ids_in_model_order) image_ids_in_model_order->push_back(image.id);
    std::istringstream points(points_line);
    points.imbue(std::locale::classic());
    double x = 0.0;
    double y = 0.0;
    std::string point3d_id;
    Point2DIdx point2d_idx = 0;
    while (points >> std::ws && !points.eof()) {
      if (!(points >> x >> y >> point3d_id)) {
        throw std::runtime_error("Invalid COLMAP POINTS2D entry: " + points_line);
      }
      image.points2d.push_back(
          {Eigen::Vector2d(x, y), point2d_idx++,
           ParsePoint3DId(point3d_id)});
    }
    reconstruction.AddImage(std::move(image));
  }

  for (const std::string& line : DataLines(model_path / "points3D.txt")) {
    std::istringstream stream(line);
    stream.imbue(std::locale::classic());
    Point3D point;
    int red = 0;
    int green = 0;
    int blue = 0;
    stream >> point.id >> point.xyz.x() >> point.xyz.y() >> point.xyz.z() >>
        red >> green >> blue >> point.reprojection_error;
    if (!stream || red < 0 || red > 255 || green < 0 || green > 255 ||
        blue < 0 || blue > 255) {
      throw std::runtime_error("Invalid COLMAP point entry: " + line);
    }
    point.color = Eigen::Matrix<std::uint8_t, 3, 1>(
        static_cast<std::uint8_t>(red), static_cast<std::uint8_t>(green),
        static_cast<std::uint8_t>(blue));
    ImageId image_id = kInvalidImageId;
    Point2DIdx point2d_idx = kInvalidPoint2DIdx;
    while (stream >> std::ws && !stream.eof()) {
      if (!(stream >> image_id >> point2d_idx)) {
        throw std::runtime_error("Invalid COLMAP track entry: " + line);
      }
      point.track.AddElement(image_id, point2d_idx);
    }
    reconstruction.AddPoint3D(std::move(point));
  }
  ApplyTrivialColmapFrames(model_path, false, &reconstruction, image_ids_in_model_order);
  return reconstruction;
}

Reconstruction ReadColmapBinaryReconstruction(
    const std::filesystem::path& model_path,
    std::vector<ImageId>* image_ids_in_model_order) {
  if (image_ids_in_model_order) image_ids_in_model_order->clear();
  std::ifstream cameras_file(model_path / "cameras.bin", std::ios::binary);
  std::ifstream images_file(model_path / "images.bin", std::ios::binary);
  std::ifstream points_file(model_path / "points3D.bin", std::ios::binary);
  if (!cameras_file || !images_file || !points_file) {
    throw std::runtime_error("Failed to open complete COLMAP binary model: " +
                             model_path.string());
  }

  Reconstruction reconstruction;
  std::uint64_t num_cameras = 0;
  ReadColmapValue(&cameras_file, &num_cameras);
  for (std::uint64_t index = 0; index < num_cameras; ++index) {
    Camera camera;
    std::int32_t model_id = -1;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    ReadColmapValue(&cameras_file, &camera.id);
    ReadColmapValue(&cameras_file, &model_id);
    ReadColmapValue(&cameras_file, &width);
    ReadColmapValue(&cameras_file, &height);
    if (width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("COLMAP binary camera dimensions overflow");
    }
    camera.model_id = static_cast<CameraModelId>(model_id);
    camera.width = static_cast<std::uint32_t>(width);
    camera.height = static_cast<std::uint32_t>(height);
    if (!ExistsCameraModel(camera.model_id)) {
      throw std::runtime_error("Unknown COLMAP binary camera model id: " +
                               std::to_string(model_id));
    }
    camera.params.resize(GetCameraModelSpec(camera.model_id).param_names.size());
    for (double& parameter : camera.params) {
      ReadColmapValue(&cameras_file, &parameter);
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid COLMAP binary camera");
    }
    reconstruction.AddCamera(std::move(camera));
  }

  std::uint64_t num_images = 0;
  ReadColmapValue(&images_file, &num_images);
  for (std::uint64_t index = 0; index < num_images; ++index) {
    Image image;
    Pose3 pose;
    ReadColmapValue(&images_file, &image.id);
    ReadColmapValue(&images_file, &pose.q_cw.w());
    ReadColmapValue(&images_file, &pose.q_cw.x());
    ReadColmapValue(&images_file, &pose.q_cw.y());
    ReadColmapValue(&images_file, &pose.q_cw.z());
    ReadColmapValue(&images_file, &pose.t_cw.x());
    ReadColmapValue(&images_file, &pose.t_cw.y());
    ReadColmapValue(&images_file, &pose.t_cw.z());
    ReadColmapValue(&images_file, &image.camera_id);
    image.name = ReadNullTerminated(&images_file);
    std::uint64_t num_points2d = 0;
    ReadColmapValue(&images_file, &num_points2d);
    if (num_points2d > std::numeric_limits<Point2DIdx>::max()) {
      throw std::runtime_error("COLMAP binary image has too many 2D points");
    }
    image.points2d.reserve(static_cast<std::size_t>(num_points2d));
    for (std::uint64_t point_index = 0; point_index < num_points2d;
         ++point_index) {
      Eigen::Vector2d xy;
      Point3DId point3d_id = kInvalidPoint3DId;
      ReadColmapValue(&images_file, &xy.x());
      ReadColmapValue(&images_file, &xy.y());
      ReadColmapValue(&images_file, &point3d_id);
      image.points2d.push_back(
          {xy, static_cast<Point2DIdx>(point_index), point3d_id});
    }
    if (!reconstruction.ExistsCamera(image.camera_id)) {
      throw std::runtime_error(
          "COLMAP binary image references a missing camera");
    }
    image.cam_from_world = pose;
    image.registered = true;
    if (image_ids_in_model_order) image_ids_in_model_order->push_back(image.id);
    reconstruction.AddImage(std::move(image));
  }

  std::uint64_t num_points3d = 0;
  ReadColmapValue(&points_file, &num_points3d);
  for (std::uint64_t index = 0; index < num_points3d; ++index) {
    Point3D point;
    ReadColmapValue(&points_file, &point.id);
    ReadColmapValue(&points_file, &point.xyz.x());
    ReadColmapValue(&points_file, &point.xyz.y());
    ReadColmapValue(&points_file, &point.xyz.z());
    ReadColmapValue(&points_file, &point.color.x());
    ReadColmapValue(&points_file, &point.color.y());
    ReadColmapValue(&points_file, &point.color.z());
    ReadColmapValue(&points_file, &point.reprojection_error);
    std::uint64_t track_length = 0;
    ReadColmapValue(&points_file, &track_length);
    point.track.elements.reserve(static_cast<std::size_t>(track_length));
    for (std::uint64_t track_index = 0; track_index < track_length;
         ++track_index) {
      TrackElement element;
      ReadColmapValue(&points_file, &element.image_id);
      ReadColmapValue(&points_file, &element.point2d_idx);
      point.track.elements.push_back(element);
    }
    reconstruction.AddPoint3D(std::move(point));
  }
  ApplyTrivialColmapFrames(model_path, true, &reconstruction, image_ids_in_model_order);
  return reconstruction;
}

Reconstruction ReadColmapReconstruction(
    const std::filesystem::path& model_path, ColmapModelFormat format,
    std::vector<ImageId>* image_ids_in_model_order) {
  if (format == ColmapModelFormat::kAuto) {
    format = DetectColmapModelFormat(model_path);
  }
  if (format == ColmapModelFormat::kText) {
    return ReadColmapTextReconstruction(model_path, image_ids_in_model_order);
  }
  if (format == ColmapModelFormat::kBinary) {
    return ReadColmapBinaryReconstruction(model_path, image_ids_in_model_order);
  }
  throw std::invalid_argument(
      "Reading COLMAP model format 'both' is ambiguous; use auto, txt, or bin");
}

Reconstruction ReadWorkspaceReconstruction(
    const std::filesystem::path& workspace_path,
    const std::filesystem::path& input_model_path) {
  const std::vector<Camera> workspace_cameras =
      ReadWorkspaceCameras(workspace_path);
  const std::vector<Image> workspace_images =
      ReadWorkspaceImages(workspace_path);
  if (input_model_path.empty()) {
    Reconstruction reconstruction;
    for (const Camera& camera : workspace_cameras) {
      reconstruction.AddCamera(camera);
    }
    for (const Image& image : workspace_images) {
      reconstruction.AddImage(image);
    }
    return reconstruction;
  }

  Reconstruction schema;
  for (const Camera& camera : workspace_cameras) {
    schema.AddCamera(camera);
  }
  for (const Image& image : workspace_images) {
    schema.AddImage(image);
  }
  return ApplyColmapModel(schema, input_model_path);
}

Reconstruction ApplyColmapTextModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path) {
  return ApplyColmapModel(schema, input_model_path,
                          ColmapModelFormat::kText);
}

Reconstruction ApplyColmapModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path,
    const ColmapModelFormat format) {
  const Reconstruction model =
      ReadColmapReconstruction(input_model_path, format);
  std::map<std::string, ImageId> schema_image_ids;
  for (const auto& [image_id, image] : schema.Images()) {
    if (!schema_image_ids.emplace(image.name, image_id).second) {
      throw std::runtime_error("Workspace contains duplicate image name: " +
                               image.name);
    }
  }
  std::map<ImageId, ImageId> model_to_schema_image;
  std::map<CameraId, Camera> cameras;
  std::map<CameraId, Camera> mapped_model_cameras;
  for (const auto& [camera_id, camera] : schema.Cameras()) {
    cameras.emplace(camera_id, camera);
  }
  for (const auto& [model_image_id, model_image] : model.Images()) {
    const auto schema_id_it = schema_image_ids.find(model_image.name);
    if (schema_id_it == schema_image_ids.end()) {
      throw std::runtime_error(
          "Input model image is absent from current image set: " +
          model_image.name);
    }
    const Image& schema_image = schema.GetImage(schema_id_it->second);
    const Camera& model_camera = model.GetCamera(model_image.camera_id);
    Camera optimized_camera = model_camera;
    optimized_camera.id = schema_image.camera_id;
    const Camera& schema_camera = schema.GetCamera(schema_image.camera_id);
    if (optimized_camera.model_id != schema_camera.model_id ||
        optimized_camera.width != schema_camera.width ||
        optimized_camera.height != schema_camera.height) {
      throw std::runtime_error(
          "Input model camera is incompatible with current image: " +
          model_image.name);
    }
    const auto [mapped_it, inserted] = mapped_model_cameras.emplace(
        schema_image.camera_id, optimized_camera);
    if (!inserted &&
        !CameraParametersEqual(mapped_it->second, optimized_camera)) {
      throw std::runtime_error(
          "Input model has incompatible intrinsics for images sharing "
          "workspace camera " +
          std::to_string(schema_image.camera_id));
    }
    cameras[schema_image.camera_id] = std::move(optimized_camera);
    model_to_schema_image.emplace(model_image_id, schema_id_it->second);
  }

  Reconstruction reconstruction;
  for (const auto& [camera_id, camera] : cameras) {
    (void)camera_id;
    reconstruction.AddCamera(camera);
  }
  std::map<ImageId, const Image*> model_images_by_schema_id;
  for (const auto& [model_image_id, schema_image_id] :
       model_to_schema_image) {
    model_images_by_schema_id.emplace(
        schema_image_id, &model.GetImage(model_image_id));
  }
  for (const auto& [image_id, schema_image] : schema.Images()) {
    (void)image_id;
    const auto model_image_it = model_images_by_schema_id.find(image_id);
    if (model_image_it == model_images_by_schema_id.end()) {
      reconstruction.AddImage(schema_image);
      continue;
    }
    Image image = *model_image_it->second;
    image.id = image_id;
    image.camera_id = schema_image.camera_id;
    image.file_path = schema_image.file_path;
    image.prior_position = schema_image.prior_position;
    reconstruction.AddImage(std::move(image));
  }
  if (reconstruction.RegisteredImageIds().size() !=
      model.RegisteredImageIds().size()) {
    throw std::runtime_error(
        "Input model contains images outside the workspace");
  }
  for (const auto& [point_id, point] : model.Points3D()) {
    (void)point_id;
    Point3D remapped = point;
    for (TrackElement& element : remapped.track.elements) {
      element.image_id = model_to_schema_image.at(element.image_id);
    }
    reconstruction.AddPoint3D(std::move(remapped));
  }
  return reconstruction;
}

}  // namespace colite
