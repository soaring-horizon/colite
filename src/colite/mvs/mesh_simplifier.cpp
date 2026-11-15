#include "colite/mvs/meshing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "colite/core/parallel.h"

namespace colite::mvs {
namespace {

constexpr double kMinAreaSquared = 1e-24;

struct VertexState {
  FusedPoint point;
  Eigen::Vector3f color = Eigen::Vector3f::Zero();
  Eigen::Matrix4d quadric = Eigen::Matrix4d::Zero();
  std::vector<int> faces;
  std::uint64_t version = 0;
  bool active = true;
};

struct FaceState {
  std::array<int, 3> vertices{};
  bool active = true;
};

struct CollapseCandidate {
  double error = 0.0;
  Eigen::Vector3f color = Eigen::Vector3f::Zero();
  int first = -1;
  int second = -1;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  std::uint64_t first_version = 0;
  std::uint64_t second_version = 0;
};

struct GreaterError {
  bool operator()(const CollapseCandidate& lhs,
                  const CollapseCandidate& rhs) const {
    if (lhs.error != rhs.error) return lhs.error > rhs.error;
    if (lhs.first != rhs.first) return lhs.first > rhs.first;
    return lhs.second > rhs.second;
  }
};

std::array<int, 2> OrderedEdge(int first, int second) {
  return first < second ? std::array<int, 2>{first, second}
                        : std::array<int, 2>{second, first};
}

bool FaceContains(const FaceState& face, int vertex) {
  return face.vertices[0] == vertex || face.vertices[1] == vertex ||
         face.vertices[2] == vertex;
}

std::vector<int> ActiveNeighbors(int vertex,
                                 const std::vector<VertexState>& vertices,
                                 const std::vector<FaceState>& faces) {
  std::unordered_set<int> unique;
  for (const int face_id : vertices[vertex].faces) {
    if (face_id < 0 || static_cast<std::size_t>(face_id) >= faces.size() ||
        !faces[face_id].active || !FaceContains(faces[face_id], vertex)) {
      continue;
    }
    for (const int other : faces[face_id].vertices) {
      if (other != vertex && vertices[other].active) {
        unique.insert(other);
      }
    }
  }
  return {unique.begin(), unique.end()};
}

int ActiveEdgeFaceCount(int first,
                        int second,
                        const std::vector<VertexState>& vertices,
                        const std::vector<FaceState>& faces) {
  int count = 0;
  for (const int face_id : vertices[first].faces) {
    if (face_id >= 0 && static_cast<std::size_t>(face_id) < faces.size() &&
        faces[face_id].active && FaceContains(faces[face_id], first) &&
        FaceContains(faces[face_id], second)) {
      ++count;
    }
  }
  return count;
}

double QuadricError(const Eigen::Matrix4d& quadric,
                    const Eigen::Vector3d& position) {
  const Eigen::Vector4d homogeneous(position.x(), position.y(), position.z(),
                                    1.0);
  const double error = homogeneous.dot(quadric * homogeneous);
  return std::isfinite(error) ? std::max(0.0, error)
                              : std::numeric_limits<double>::infinity();
}

CollapseCandidate MakeCandidate(int first, int second,
                                const std::vector<VertexState>& vertices,
                                bool interpolate_colors) {
  const auto edge = OrderedEdge(first, second);
  first = edge[0];
  second = edge[1];
  const auto& a = vertices[first];
  const auto& b = vertices[second];
  const Eigen::Matrix4d quadric = a.quadric + b.quadric;
  const double error_a = QuadricError(quadric, a.point.xyz);
  const double error_b = QuadricError(quadric, b.point.xyz);
  CollapseCandidate candidate;
  candidate.first = first;
  candidate.second = second;
  candidate.first_version = a.version;
  candidate.second_version = b.version;
  candidate.position = error_a <= error_b ? a.point.xyz : b.point.xyz;
  candidate.error = std::min(error_a, error_b);
  const Eigen::Vector3d midpoint = 0.5 * (a.point.xyz + b.point.xyz);
  const double error_mid = QuadricError(quadric, midpoint);
  if (error_mid < candidate.error) {
    candidate.position = midpoint;
    candidate.error = error_mid;
  }
  // Use the full spatial minimizer. Restricting it to the edge segment changes
  // the QEM objective and can discard a substantially better surface fit.
  const Eigen::Matrix3d system = quadric.topLeftCorner<3, 3>();
  if (std::abs(system.determinant()) > 1e-12) {
    const Eigen::LDLT<Eigen::Matrix3d> solver(system);
    const Eigen::Vector3d position = solver.solve(-quadric.topRightCorner<3, 1>());
    if (solver.info() == Eigen::Success && position.allFinite()) {
      const double error = QuadricError(quadric, position);
      if (std::isfinite(error)) {
        candidate.position = position;
        candidate.error = error;
      }
    }
  }
  if (interpolate_colors) {
    const Eigen::Vector3d direction = b.point.xyz - a.point.xyz;
    const double length_squared = direction.squaredNorm();
    const float t = length_squared > 0
        ? static_cast<float>(std::clamp((candidate.position - a.point.xyz).dot(direction) / length_squared, 0.0, 1.0))
        : 0.5F;
    candidate.color = (1.0F - t) * a.color + t * b.color;
  } else {
    candidate.color = error_a <= error_b ? a.color : b.color;
  }
  return candidate;
}

bool SatisfiesLinkCondition(int first,
                            int second,
                            const std::vector<VertexState>& vertices,
                            const std::vector<FaceState>& faces) {
  const int edge_faces = ActiveEdgeFaceCount(first, second, vertices, faces);
  if (edge_faces < 1 || edge_faces > 2) {
    return false;
  }
  const std::vector<int> first_neighbors =
      ActiveNeighbors(first, vertices, faces);
  const std::vector<int> second_neighbors =
      ActiveNeighbors(second, vertices, faces);
  std::unordered_set<int> first_set(first_neighbors.begin(),
                                    first_neighbors.end());
  int common = 0;
  for (const int neighbor : second_neighbors) {
    if (neighbor != first && neighbor != second &&
        first_set.count(neighbor) != 0) {
      ++common;
    }
  }
  if (common != edge_faces) return false;
  // Compare the edges of the vertex links too. Vertex-count-only checks miss
  // a tetrahedron, whose collapse creates two copies of the same triangle.
  const auto link_edges = [&](int vertex) {
    std::set<std::array<int, 2>> result;
    for (const auto face_id : vertices[vertex].faces) {
      const auto& face = faces[face_id];
      if (!face.active || !FaceContains(face, vertex)) continue;
      std::array<int, 2> opposite;
      int count = 0;
      for (const auto id : face.vertices) if (id != vertex) opposite[count++] = id;
      result.insert(OrderedEdge(opposite[0], opposite[1]));
    }
    return result;
  };
  const auto first_links = link_edges(first);
  for (const auto& edge : link_edges(second)) {
    if (first_links.count(edge)) return false;
  }
  const auto boundary_vertex = [&](int vertex, const std::vector<int>& neighbors) {
    for (const auto neighbor : neighbors) {
      if (ActiveEdgeFaceCount(vertex, neighbor, vertices, faces) == 1) return true;
    }
    return false;
  };
  // An interior edge between two boundary vertices must not join separate
  // boundary arcs or collapse the last two faces of an open component.
  return edge_faces == 1 || !boundary_vertex(first, first_neighbors) ||
                           !boundary_vertex(second, second_neighbors);
}

bool PreservesFaceNormals(const CollapseCandidate& candidate,
                          double minimum_normal_dot,
                          const std::vector<VertexState>& vertices,
                          const std::vector<FaceState>& faces) {
  std::unordered_set<int> checked_faces;
  for (const int endpoint : {candidate.first, candidate.second}) {
    for (const int face_id : vertices[endpoint].faces) {
      if (!checked_faces.insert(face_id).second || face_id < 0 ||
          static_cast<std::size_t>(face_id) >= faces.size() ||
          !faces[face_id].active) {
        continue;
      }
      const FaceState& face = faces[face_id];
      if (FaceContains(face, candidate.first) &&
          FaceContains(face, candidate.second)) {
        continue;
      }
      std::array<Eigen::Vector3d, 3> old_points;
      std::array<Eigen::Vector3d, 3> new_points;
      for (int corner = 0; corner < 3; ++corner) {
        old_points[corner] = vertices[face.vertices[corner]].point.xyz;
        new_points[corner] =
            face.vertices[corner] == candidate.first ||
                    face.vertices[corner] == candidate.second
                ? candidate.position
                : old_points[corner];
      }
      const Eigen::Vector3d old_cross =
          (old_points[1] - old_points[0]).cross(old_points[2] - old_points[0]);
      const Eigen::Vector3d new_cross =
          (new_points[1] - new_points[0]).cross(new_points[2] - new_points[0]);
      if (new_cross.squaredNorm() <= kMinAreaSquared ||
          old_cross.squaredNorm() <= kMinAreaSquared ||
          old_cross.normalized().dot(new_cross.normalized()) <
              minimum_normal_dot) {
        return false;
      }
    }
  }
  return true;
}

bool CanCollapse(const CollapseCandidate& candidate,
                 const std::vector<VertexState>& vertices,
                 const std::vector<FaceState>& faces) {
  if (!std::isfinite(candidate.error) || !vertices[candidate.first].active ||
      !vertices[candidate.second].active ||
      candidate.first_version != vertices[candidate.first].version ||
      candidate.second_version != vertices[candidate.second].version ||
      !SatisfiesLinkCondition(candidate.first, candidate.second, vertices,
                              faces)) {
    return false;
  }
  return PreservesFaceNormals(candidate, 0.0, vertices, faces);
}

std::size_t Collapse(const CollapseCandidate& candidate,
                     std::vector<VertexState>* vertices,
                     std::vector<FaceState>* faces) {
  VertexState& kept = (*vertices)[candidate.first];
  VertexState& removed = (*vertices)[candidate.second];
  kept.point.xyz = candidate.position;
  kept.color = candidate.color;
  kept.quadric += removed.quadric;
  ++kept.version;
  removed.active = false;
  ++removed.version;

  kept.faces.insert(kept.faces.end(), removed.faces.begin(),
                    removed.faces.end());
  std::sort(kept.faces.begin(), kept.faces.end());
  kept.faces.erase(std::unique(kept.faces.begin(), kept.faces.end()),
                   kept.faces.end());
  std::size_t removed_faces = 0;
  for (const int face_id : kept.faces) {
    FaceState& face = (*faces)[face_id];
    if (!face.active) {
      continue;
    }
    for (int& vertex : face.vertices) {
      if (vertex == candidate.second) {
        vertex = candidate.first;
      }
    }
    if (face.vertices[0] == face.vertices[1] ||
        face.vertices[1] == face.vertices[2] ||
        face.vertices[2] == face.vertices[0]) {
      face.active = false;
      ++removed_faces;
    }
  }
  return removed_faces;
}

void RecomputeNormals(DenseTriangleMesh* mesh) {
  for (FusedPoint& vertex : mesh->vertices) {
    vertex.normal.setZero();
  }
  for (const Eigen::Vector3i& triangle : mesh->triangles) {
    const Eigen::Vector3d cross =
        (mesh->vertices[triangle.y()].xyz -
         mesh->vertices[triangle.x()].xyz)
            .cross(mesh->vertices[triangle.z()].xyz -
                   mesh->vertices[triangle.x()].xyz);
    for (int corner = 0; corner < 3; ++corner) {
      mesh->vertices[triangle[corner]].normal += cross.cast<float>();
    }
  }
  for (FusedPoint& vertex : mesh->vertices) {
    if (vertex.normal.squaredNorm() > 0.0F) {
      vertex.normal.normalize();
    }
  }
}

}  // namespace

bool MeshSimplificationOptions::Check() const {
  return std::isfinite(target_face_ratio) && target_face_ratio > 0.0 &&
         target_face_ratio <= 1.0 && std::isfinite(max_error) && max_error >= 0.0 &&
         std::isfinite(boundary_weight) && boundary_weight >= 0.0 &&
         (num_threads == -1 || num_threads > 0);
}

DenseTriangleMesh SimplifyMesh(const DenseTriangleMesh& mesh,
                               const MeshSimplificationOptions& options) {
  if (!options.Check()) throw std::invalid_argument("Invalid mesh simplification options.");
  if (mesh.vertices.empty() || mesh.triangles.empty()) return mesh;
  if (mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      mesh.triangles.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Mesh exceeds the supported index range");
  }
  for (const auto& face : mesh.triangles) {
    if (face.minCoeff() < 0 || face.maxCoeff() >= static_cast<int>(mesh.vertices.size())) {
      throw std::invalid_argument("Mesh face index is outside the vertex array");
    }
  }
  // A no-op ratio preserves vertex attributes and unreferenced vertices too.
  if (options.target_face_ratio == 1.0) return mesh;
  struct ThreadBudget {
    int previous = parallel::getNumThreads();
    explicit ThreadBudget(int count) { parallel::setNumThreads(count); }
    ~ThreadBudget() { parallel::setNumThreads(previous); }
  } budget(options.num_threads);

  std::vector<VertexState> vertices(mesh.vertices.size());
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    vertices[i].point = mesh.vertices[i];
    vertices[i].color = mesh.vertices[i].color.cast<float>();
  }
  std::vector<Eigen::Matrix4d> face_quadrics(mesh.triangles.size(), Eigen::Matrix4d::Zero());
  std::vector<std::uint8_t> valid(mesh.triangles.size(), 0);
  parallel::parallel_for_(parallel::Range(0, static_cast<int>(mesh.triangles.size())),
      [&](const parallel::Range& range) {
    for (int i = range.start; i < range.end; ++i) {
      const auto& face = mesh.triangles[i];
      if (face.x() == face.y() || face.y() == face.z() || face.z() == face.x()) continue;
      const auto& a = mesh.vertices[face.x()].xyz;
      Eigen::Vector3d normal = (mesh.vertices[face.y()].xyz - a).cross(mesh.vertices[face.z()].xyz - a);
      if (!normal.allFinite() || normal.squaredNorm() <= kMinAreaSquared) continue;
      normal.normalize();
      const Eigen::Vector4d plane(normal.x(), normal.y(), normal.z(), -normal.dot(a));
      face_quadrics[i] = plane * plane.transpose();
      valid[i] = 1;
    }
  });
  std::vector<FaceState> faces;
  faces.reserve(mesh.triangles.size());
  // Reduce in input order, independently of the initialization thread budget.
  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    if (!valid[i]) continue;
    const auto& face = mesh.triangles[i];
    const int face_id = static_cast<int>(faces.size());
    faces.push_back({{face.x(), face.y(), face.z()}, true});
    for (int corner = 0; corner < 3; ++corner) {
      vertices[face[corner]].quadric += face_quadrics[i];
      vertices[face[corner]].faces.push_back(face_id);
    }
  }
  face_quadrics.clear();
  face_quadrics.shrink_to_fit();
  struct EdgeInfo { int count = 0; int face = -1; };
  std::map<std::array<int, 2>, EdgeInfo> edges;
  for (std::size_t i = 0; i < faces.size(); ++i) {
    const auto& face = faces[i];
    for (int j = 0; j < 3; ++j) {
      auto& info = edges[OrderedEdge(face.vertices[j], face.vertices[(j + 1) % 3])];
      ++info.count;
      info.face = static_cast<int>(i);
    }
  }
  if (options.boundary_weight > 0) {
    for (const auto& [edge, info] : edges) {
      if (info.count != 1) continue;
      const auto& face = faces[info.face];
      const auto& a = vertices[face.vertices[0]].point.xyz;
      const Eigen::Vector3d normal = (vertices[face.vertices[1]].point.xyz - a).cross(
          vertices[face.vertices[2]].point.xyz - a);
      const auto& origin = vertices[edge[0]].point.xyz;
      const Eigen::Vector3d direction = vertices[edge[1]].point.xyz - origin;
      Eigen::Vector3d boundary_normal = direction.cross(normal);
      if (boundary_normal.norm() < 1e-12) continue;
      boundary_normal.normalize();
      const Eigen::Vector4d plane(boundary_normal.x(), boundary_normal.y(), boundary_normal.z(), -boundary_normal.dot(origin));
      const Eigen::Matrix4d quadric = options.boundary_weight * plane * plane.transpose();
      vertices[edge[0]].quadric += quadric;
      vertices[edge[1]].quadric += quadric;
    }
  }
  std::vector<std::array<int, 2>> edge_ids;
  edge_ids.reserve(edges.size());
  for (const auto& [edge, info] : edges) edge_ids.push_back(edge);
  if (edge_ids.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Mesh exceeds the supported edge range");
  }
  std::vector<CollapseCandidate> candidates(edge_ids.size());
  parallel::parallel_for_(parallel::Range(0, static_cast<int>(edge_ids.size())), [&](const parallel::Range& range) {
    for (int i = range.start; i < range.end; ++i) {
      candidates[i] = MakeCandidate(edge_ids[i][0], edge_ids[i][1], vertices, options.interpolate_colors);
    }
  });
  std::priority_queue<CollapseCandidate, std::vector<CollapseCandidate>, GreaterError> queue(
      GreaterError{}, std::move(candidates));
  const std::size_t target_faces = std::max<std::size_t>(1, static_cast<std::size_t>(
      std::floor(static_cast<double>(mesh.triangles.size()) * options.target_face_ratio)));
  std::size_t active_faces = faces.size();
  while (active_faces > target_faces && !queue.empty()) {
    const auto candidate = queue.top();
    queue.pop();
    if (!vertices[candidate.first].active || !vertices[candidate.second].active ||
        candidate.first_version != vertices[candidate.first].version ||
        candidate.second_version != vertices[candidate.second].version) continue;
    if (options.max_error > 0 && candidate.error > options.max_error) break;
    if (!CanCollapse(candidate, vertices, faces)) continue;
    active_faces -= Collapse(candidate, &vertices, &faces);
    for (const auto neighbor : ActiveNeighbors(candidate.first, vertices, faces)) {
      queue.push(MakeCandidate(candidate.first, neighbor, vertices, options.interpolate_colors));
    }
  }
  DenseTriangleMesh result;
  std::vector<int> new_indices(vertices.size(), -1);
  std::set<std::array<int, 3>> unique_faces;
  for (const auto& face : faces) {
    if (!face.active) continue;
    auto canonical = face.vertices;
    std::sort(canonical.begin(), canonical.end());
    if (!unique_faces.insert(canonical).second) continue;
    Eigen::Vector3i triangle;
    for (int corner = 0; corner < 3; ++corner) {
      const int old_index = face.vertices[corner];
      if (new_indices[old_index] < 0) {
        new_indices[old_index] = static_cast<int>(result.vertices.size());
        auto point = vertices[old_index].point;
        for (int channel = 0; channel < 3; ++channel) {
          point.color[channel] = static_cast<std::uint8_t>(std::clamp(std::round(vertices[old_index].color[channel]), 0.0F, 255.0F));
        }
        result.vertices.push_back(point);
      }
      triangle[corner] = new_indices[old_index];
    }
    result.triangles.push_back(triangle);
  }
  RecomputeNormals(&result);
  return result;
}

DenseTriangleMesh ReadDenseTriangleMeshPly(
    const std::filesystem::path& path,
    std::vector<Eigen::Vector2f>* face_uvs,
    std::string* texture_file) {
  if (face_uvs) face_uvs->clear();
  if (texture_file) texture_file->clear();
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read triangle mesh: " + path.string());
  }
  std::string line;
  if (!std::getline(file, line)) {
    throw std::runtime_error("Invalid PLY header: " + path.string());
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != "ply") {
    throw std::runtime_error("Invalid PLY header: " + path.string());
  }
  std::size_t num_vertices = 0;
  std::size_t num_faces = 0;
  struct Property {
    bool list = false;
    std::string count_type;
    std::string value_type;
    std::string name;
  };
  std::vector<Property> vertex_properties;
  std::vector<Property> face_properties;
  std::string element;
  bool ascii = false;
  bool binary_little_endian = false;
  bool binary_big_endian = false;
  bool end_header = false;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string keyword;
    stream >> keyword;
    if (keyword == "comment") {
      std::string label;
      stream >> label;
      if (label == "TextureFile" && texture_file) {
        std::getline(stream >> std::ws, *texture_file);
        if (!texture_file->empty() && texture_file->back() == '\r') texture_file->pop_back();
      }
    } else if (keyword == "format") {
      std::string format;
      stream >> format;
      ascii = format == "ascii";
      binary_little_endian = format == "binary_little_endian";
      binary_big_endian = format == "binary_big_endian";
    } else if (keyword == "element") {
      std::size_t count = 0;
      stream >> element >> count;
      if (element == "vertex") {
        num_vertices = count;
      } else if (element == "face") {
        num_faces = count;
      }
    } else if (keyword == "property" &&
               (element == "vertex" || element == "face")) {
      Property property;
      stream >> property.value_type;
      if (property.value_type == "list") {
        property.list = true;
        stream >> property.count_type >> property.value_type >> property.name;
      } else {
        stream >> property.name;
      }
      (element == "vertex" ? vertex_properties : face_properties)
          .push_back(std::move(property));
    } else if (keyword == "end_header") {
      end_header = true;
      break;
    }
  }
  if ((!ascii && !binary_little_endian && !binary_big_endian) || !end_header) {
    throw std::runtime_error("Unsupported PLY encoding.");
  }
  const auto type_size = [](const std::string& type) -> std::size_t {
    if (type == "char" || type == "int8" || type == "uchar" ||
        type == "uint8") return 1;
    if (type == "short" || type == "int16" || type == "ushort" ||
        type == "uint16") return 2;
    if (type == "int" || type == "int32" || type == "uint" ||
        type == "uint32" || type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64" || type == "int64" ||
        type == "uint64") return 8;
    throw std::runtime_error("Unsupported PLY property type: " + type);
  };
  const bool swap_bytes = binary_big_endian;
  const auto read_binary = [&](const std::string& type) -> double {
    std::array<unsigned char, 8> bytes{};
    const std::size_t size = type_size(type);
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Unexpected end of binary PLY data.");
    if (swap_bytes) std::reverse(bytes.begin(), bytes.begin() + size);
    if (type == "char" || type == "int8") {
      std::int8_t value = 0; std::memcpy(&value, bytes.data(), 1); return value;
    }
    if (type == "uchar" || type == "uint8") return bytes[0];
    if (type == "short" || type == "int16") {
      std::int16_t value = 0; std::memcpy(&value, bytes.data(), 2); return value;
    }
    if (type == "ushort" || type == "uint16") {
      std::uint16_t value = 0; std::memcpy(&value, bytes.data(), 2); return value;
    }
    if (type == "int" || type == "int32") {
      std::int32_t value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "uint" || type == "uint32") {
      std::uint32_t value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "float" || type == "float32") {
      float value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "double" || type == "float64") {
      double value = 0; std::memcpy(&value, bytes.data(), 8); return value;
    }
    if (type == "int64") {
      std::int64_t value = 0; std::memcpy(&value, bytes.data(), 8);
      return static_cast<double>(value);
    }
    std::uint64_t value = 0; std::memcpy(&value, bytes.data(), 8);
    return static_cast<double>(value);
  };
  DenseTriangleMesh mesh;
  mesh.vertices.resize(num_vertices);
  for (FusedPoint& vertex : mesh.vertices) {
    std::istringstream stream;
    if (ascii) {
      if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of PLY vertex data.");
      }
      stream.str(line);
    }
    for (const Property& property : vertex_properties) {
      if (property.list) {
        throw std::runtime_error("Unsupported vertex list property in PLY.");
      }
      double value = 0.0;
      if (ascii) {
        if (!(stream >> value)) throw std::runtime_error("Invalid PLY vertex data.");
      } else {
        value = read_binary(property.value_type);
      }
      if (property.name == "x") vertex.xyz.x() = value;
      else if (property.name == "y") vertex.xyz.y() = value;
      else if (property.name == "z") vertex.xyz.z() = value;
      else if (property.name == "nx") vertex.normal.x() = static_cast<float>(value);
      else if (property.name == "ny") vertex.normal.y() = static_cast<float>(value);
      else if (property.name == "nz") vertex.normal.z() = static_cast<float>(value);
      else if (property.name == "red")
        vertex.color.x() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
      else if (property.name == "green")
        vertex.color.y() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
      else if (property.name == "blue")
        vertex.color.z() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
    }
  }
  mesh.triangles.reserve(num_faces);
  for (std::size_t i = 0; i < num_faces; ++i) {
    std::istringstream stream;
    if (ascii) {
      if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of PLY face data.");
      }
      stream.str(line);
    }
    std::vector<int> indices;
    std::vector<Eigen::Vector2f> uvs;
    for (const Property& property : face_properties) {
      if (!property.list) {
        double ignored = 0.0;
        if (ascii) stream >> ignored;
        else ignored = read_binary(property.value_type);
        (void)ignored;
        continue;
      }
      std::size_t count = 0;
      if (ascii) stream >> count;
      else count = static_cast<std::size_t>(read_binary(property.count_type));
      const bool is_indices = property.name == "vertex_indices" || property.name == "vertex_index";
      const bool is_uv = property.name == "texcoord";
      if ((is_indices && count != 3) || (is_uv && count != 6) ||
          count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid PLY face list length");
      }
      std::vector<double> values(count);
      for (double& value : values) {
        if (ascii) stream >> value;
        else value = read_binary(property.value_type);
        if (!std::isfinite(value) || (ascii && !stream)) throw std::runtime_error("Invalid PLY face value");
      }
      if (is_indices) {
        for (const auto value : values) {
          if (value < 0 || value >= num_vertices || value > std::numeric_limits<int>::max() || std::floor(value) != value) {
            throw std::runtime_error("Invalid PLY vertex index");
          }
          indices.push_back(static_cast<int>(value));
        }
      } else if (is_uv) {
        for (int corner = 0; corner < 3; ++corner) {
          const Eigen::Vector2f uv(static_cast<float>(values[corner * 2]), static_cast<float>(values[corner * 2 + 1]));
          if (!uv.allFinite()) throw std::runtime_error("Invalid PLY texture coordinate");
          uvs.push_back(uv);
        }
      }
    }
    if (indices.size() == 3 && (!ascii || stream)) {
      mesh.triangles.emplace_back(indices[0], indices[1], indices[2]);
      if (face_uvs) face_uvs->insert(face_uvs->end(), uvs.begin(), uvs.end());
    } else {
      throw std::runtime_error("Invalid triangular PLY face");
    }
  }
  return mesh;
}

}  // namespace colite::mvs
