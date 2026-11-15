#include "colite/mvs/mesh_texturer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

#include "colite/mvs/meshing.h"

namespace colite::mvs {

// Optional auxiliary export, implemented alongside texture baking.
void WriteTexturedObj(const std::filesystem::path& directory,
                      const TexturedTriangleMesh& mesh);

TexturedMeshOutputType TexturedMeshOutputTypeFromString(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  if (name == "BIN") return TexturedMeshOutputType::kBinary;
  if (name == "TXT") return TexturedMeshOutputType::kText;
  throw std::invalid_argument("output_type must be BIN or TXT");
}

void WriteTexturedTriangleMesh(const std::filesystem::path& directory,
                               const TexturedTriangleMesh& mesh,
                               TexturedMeshOutputType type,
                               bool write_obj) {
  if (type != TexturedMeshOutputType::kBinary && type != TexturedMeshOutputType::kText) {
    throw std::invalid_argument("Invalid textured mesh output type");
  }
  if (mesh.mesh.triangles.size() > std::numeric_limits<std::size_t>::max() / 3 ||
      mesh.face_uvs.size() != mesh.mesh.triangles.size() * 3 ||
      mesh.mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("Textured mesh has invalid dimensions");
  }
  for (const auto& point : mesh.mesh.vertices) {
    if (!point.xyz.allFinite() || !point.xyz.cast<float>().allFinite()) {
      throw std::invalid_argument("Textured PLY requires finite float coordinates");
    }
  }
  for (const auto& face : mesh.mesh.triangles) {
    if (face.minCoeff() < 0 || face.maxCoeff() >= static_cast<int>(mesh.mesh.vertices.size())) {
      throw std::invalid_argument("Invalid textured PLY face index");
    }
  }
  for (const auto& uv : mesh.face_uvs) {
    if (!uv.allFinite()) throw std::invalid_argument("Textured PLY requires finite UV coordinates");
  }
  if (mesh.texture_atlas.empty()) throw std::invalid_argument("Texture atlas is empty");
  std::filesystem::create_directories(directory);
  std::ofstream stream(directory / "mesh.ply", std::ios::binary);
  if (!stream) throw std::runtime_error("Failed to write textured PLY");
  const bool binary = type == TexturedMeshOutputType::kBinary;
  stream << "ply\nformat " << (binary ? "binary_little_endian" : "ascii")
         << " 1.0\ncomment TextureFile texture.png\nelement vertex " << mesh.mesh.vertices.size()
         << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << mesh.mesh.triangles.size()
         << "\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n";
  stream << std::setprecision(std::numeric_limits<float>::max_digits10);
  const auto write_u32 = [&](std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) stream.put(static_cast<char>((value >> (byte * 8)) & 255));
  };
  const auto write_float = [&](float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(bits);
  };
  for (const auto& point : mesh.mesh.vertices) {
    for (int axis = 0; axis < 3; ++axis) {
      const float value = static_cast<float>(point.xyz[axis]);
      if (binary) write_float(value);
      else stream << value << (axis == 2 ? '\n' : ' ');
    }
  }
  for (std::size_t i = 0; i < mesh.mesh.triangles.size(); ++i) {
    if (binary) stream.put(3);
    else stream << '3';
    for (const auto id : mesh.mesh.triangles[i]) {
      if (binary) write_u32(static_cast<std::uint32_t>(id));
      else stream << ' ' << id;
    }
    if (binary) stream.put(6);
    else stream << " 6";
    for (int corner = 0; corner < 3; ++corner) {
      const auto& uv = mesh.face_uvs[i * 3 + corner];
      for (int axis = 0; axis < 2; ++axis) {
        if (binary) write_float(uv[axis]);
        else stream << ' ' << uv[axis];
      }
    }
    if (!binary) stream << '\n';
  }
  stream.close();
  if (!stream) throw std::runtime_error("Failed to finish textured PLY");
  if (!cv::imwrite((directory / "texture.png").string(), mesh.texture_atlas)) {
    throw std::runtime_error("Failed to write texture atlas");
  }
  if (write_obj) WriteTexturedObj(directory, mesh);
}

TexturedTriangleMesh ReadTexturedTriangleMesh(const std::filesystem::path& path) {
  TexturedTriangleMesh result;
  std::string texture_file;
  result.mesh = ReadDenseTriangleMeshPly(path, &result.face_uvs, &texture_file);
  if (result.face_uvs.size() != result.mesh.triangles.size() * 3 || texture_file.empty()) {
    throw std::runtime_error("PLY is missing texture coordinates or TextureFile metadata");
  }
  result.texture_atlas = cv::imread((path.parent_path() / texture_file).string(), cv::IMREAD_COLOR);
  if (result.texture_atlas.empty()) throw std::runtime_error("Failed to read PLY texture atlas");
  return result;
}

}  // namespace colite::mvs
