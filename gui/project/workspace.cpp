#include "workspace.h"

#include <QDataStream>
#include <QDateTime>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QQuaternion>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>
#include <QtEndian>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace colite::gui {
QString fileRevision(const QString& path) {
  const QFileInfo info(path);
  return QString::number(info.size()) + ':' + QString::number(info.lastModified().toMSecsSinceEpoch()) + ':' + QString::number(info.exists());
}
namespace {
[[noreturn]] void fail(const QString& text) {
  throw std::runtime_error(text.toUtf8().constData());
}
QFile* open(QFile& file) {
  if (!file.open(QIODevice::ReadOnly))
    fail(file.fileName() + ": " + file.errorString());
  return &file;
}
quint64 number(const QString& text) {
  bool ok = false;
  auto n = text.toULongLong(&ok);
  if (!ok) fail("Invalid unsigned number: " + text);
  return n;
}
double real(const QString& text) {
  bool ok = false;
  auto n = text.toDouble(&ok);
  if (!ok || !std::isfinite(n)) fail("Invalid finite number: " + text);
  return n;
}
QStringList fields(const QString& text) {
  static thread_local const QRegularExpression whitespace(QStringLiteral("\\s+"));
  return text.split(whitespace, Qt::SkipEmptyParts);
}
quint64 key(quint32 a, quint32 b) { return (quint64(a) << 32) | b; }
void streamOk(const QDataStream& s) {
  if (s.status() != QDataStream::Ok) fail("Truncated binary data");
}
}  // namespace
Workspace Workspace::read(const Project& project) {
  Workspace w;
  w.root = project.workspace;
  QFile manifest(QDir(w.root).filePath("images.tsv"));
  QSet<quint32> ids;
  QSet<QString> names;
  if (!w.root.isEmpty() && manifest.exists()) {
    QTextStream text(open(manifest));
    text.setCodec("UTF-8");
    while (!text.atEnd()) {
      const auto line = text.readLine();
      if (line.isEmpty() || line.startsWith('#')) continue;
      const auto row = line.split('\t');
      if (row.size() < 4) fail("Malformed images.tsv row");
      ImageRecord image;
      auto id = number(row[0]), camera = number(row[1]);
      if (id >= 0xffffffffULL || camera >= 0xffffffffULL || row[2].isEmpty() ||
          ids.contains(quint32(id)) || names.contains(row[2]))
        fail("Invalid/duplicate image record");
      image.id = quint32(id);
      image.camera = quint32(camera);
      image.name = row[2];
      // Logical image names resolve under the explicitly selected image root.
      image.path = QDir(project.imageRoot).absoluteFilePath(image.name);
      if (!QFileInfo::exists(image.path)) image.path = localPath(row[3]);
      ids.insert(image.id);
      names.insert(image.name);
      w.images.push_back(image);
    }
  } else if (!project.photoFiles.isEmpty()) {
    quint32 id = 1;
    for (auto it = project.photoFiles.cbegin(); it != project.photoFiles.cend(); ++it)
      w.images.push_back({id++, 0, it.key(), it.value()});
  } else if (!project.imageRoot.isEmpty() && QDir(project.imageRoot).exists()) {
    QStringList files;
    QDirIterator it(project.imageRoot, QDir::Files,
                    QDirIterator::Subdirectories);
    // Input identity must not change with the process's Qt plugin search path.
    // These are the desktop's documented import formats on both platforms;
    // decoding errors are reported by the reader/worker, not by omitting files.
    const QStringList formats{"jpg", "jpeg", "png"};
    while (it.hasNext()) {
      const auto path = it.next();
      if (formats.contains(QFileInfo(path).suffix().toLower()))
        files.push_back(path);
    }
    files.sort();
    quint32 id = 1;
    for (const auto& path : files)
      w.images.push_back(
          {id++, 0, QDir(project.imageRoot).relativeFilePath(path), path});
  }
  return w;
}
QVector<Keypoint> Workspace::features(quint32 image, const std::function<bool()>& cancelled) const {
  if (root.isEmpty()) return {};
  QFile file(QDir(root).filePath(QString("features/%1.bin").arg(image)));
  if (!file.exists()) return {};
  open(file);
  if (file.size() > qint64(8) * 1024 * 1024 * 1024)
    fail("Feature payload exceeds supported bound");
  QDataStream in(&file);
  in.setByteOrder(QDataStream::LittleEndian);
  in.setFloatingPointPrecision(QDataStream::SinglePrecision);
  quint32 magic = 0, version = 0, extractor = 1, scalar = 1, metric = 1,
          normalization = 2, columns = 0, reserved = 0;
  quint64 count = 0, rows = 0, cols = 0, bytes = 0, expected = 0;
  in >> magic >> version;
  if (magic != 0x434c4654 || (version < 1 || version > 3))
    fail("Unsupported feature format");
  if (version >= 2) in >> extractor >> scalar >> metric >> normalization;
  in >> count >> rows >> cols;
  if (version >= 2)
    in >> bytes >> expected;
  else
    bytes = rows * cols * 4;
  if (version == 3) in >> columns >> reserved;
  streamOk(in);
  const quint64 shape = columns > 2 ? columns - 2 : 0;
  if (extractor < 1 || extractor > 3 || scalar != 1 || metric < 1 ||
      metric > 2 || normalization < 1 || normalization > 3 || reserved ||
      !(columns == 0 || columns == 2 || columns == 4 || columns == 6) ||
      count > 10000000 || cols > 65536 || (rows != 0 && rows != count) ||
      bytes != rows * cols * 4 ||
      quint64(file.size() - file.pos()) != count * (5 + shape) * 4 + bytes)
    fail("Invalid feature metadata/length");
  const auto payloadOffset = file.pos();
  quint64 hash = 14695981039346656037ULL;
  auto hashBytes = [&](const QByteArray& block) {
    for (unsigned char c : block) {
      hash ^= c;
      hash *= 1099511628211ULL;
    }
  };
  if (version == 3) {
    quint32 little = qToLittleEndian(columns);
    hashBytes(QByteArray(reinterpret_cast<char*>(&little), 4));
  }
  // Stream the checksum, including descriptors, without retaining descriptor
  // memory.
  while (!file.atEnd()) {
    if (cancelled && cancelled()) fail("Inspection cancelled");
    const auto block = file.read(1024 * 1024);
    if (block.isEmpty() && file.error() != QFile::NoError)
      fail(file.errorString());
    hashBytes(block);
  }
  if (version >= 2 && hash != expected) fail("Feature checksum mismatch");
  file.seek(payloadOffset);
  in.resetStatus();
  QVector<Keypoint> result;
  result.reserve(int(count));
  for (quint64 i = 0; i < count; ++i) {
    if (i % 8192 == 0 && cancelled && cancelled()) fail("Inspection cancelled");
    Keypoint p;
    in >> p.x >> p.y >> p.scale >> p.orientation >> p.response;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.scale) ||
        !std::isfinite(p.orientation) || !std::isfinite(p.response))
      fail("Non-finite keypoint");
    result.push_back(p);
  }
  streamOk(in);
  return result;
}
PairData Workspace::pair(quint32 a, quint32 b, int countA, int countB,
                         const std::function<bool()>& cancelled) const {
  PairData result;
  result.a = a;
  result.b = b;
  if (root.isEmpty() || a == b) return result;
  QMap<quint64, Correspondence> matches;
  bool modern = false;
  QStringList metadata;
  quint64 declared = 0, seen = 0;
  bool empty = false;
  QSet<quint64> verifiedIndices;
  for (int verified = 0; verified < 2; ++verified) {
    QFile file(QDir(root).filePath(verified ? "matches/verified_matches.tsv"
                                            : "matches/raw_matches.tsv"));
    if (!file.exists()) continue;
    QTextStream text(open(file));
    text.setCodec("UTF-8");
    int lineNumber = 0;
    while (!text.atEnd()) {
      if (++lineNumber % 1024 == 0 && cancelled && cancelled()) fail("Inspection cancelled");
      const auto line = text.readLine();
      if (line.isEmpty()) continue;
      if (line.startsWith("# COLITE_VERIFIED_GEOMETRIES")) {
        if (line != "# COLITE_VERIFIED_GEOMETRIES 2")
          fail("Unsupported verified geometry schema");
        modern = true;
      }
      if (line.startsWith("# pair ")) {
        const auto row = fields(line.mid(7));
        if (!modern || row.size() != 40)
          fail("Malformed verified geometry metadata");
        auto x = number(row[0]), y = number(row[1]);
        if ((x == a && y == b) || (x == b && y == a)) {
          if (!metadata.isEmpty() || x >= y || number(row[2]) > 9 ||
              number(row[3]) > 15)
            fail("Invalid/duplicate pair geometry");
          for (int i = 4; i < row.size(); ++i) real(row[i]);
          metadata = row;
          declared = number(row[5]);
          result.verified = true;
        }
      }
      if (line.startsWith('#')) continue;
      const auto row = line.split('\t');
      if (row.size() != (verified ? 13 : 4))
        fail("Malformed correspondence row");
      auto x = number(row[0]), y = number(row[1]);
      if (!((x == a && y == b) || (x == b && y == a))) continue;
      if (verified && (!modern || metadata.isEmpty()))
        fail(QString::fromUtf8("匹配文件缺少可靠几何元数据，请重新执行匹配。"));
      if (verified) {
        if (number(row[2]) != number(metadata[2]) ||
            real(row[3]) != real(metadata[4]))
          fail("Inconsistent pair geometry prefix");
        for (int n = 0; n < 7; ++n)
          if (real(row[n + 4]) != real(metadata[n + 33]))
            fail("Inconsistent relative pose prefix");
      }
      const auto first = row[row.size() - 2], second = row.last();
      if (first == "-1" && second == "-1") {
        if (verified) {
          if (declared || empty) fail("Invalid empty pair sentinel");
          empty = true;
        }
        continue;
      }
      auto i = number(first), j = number(second);
      if (x != a) std::swap(i, j);
      if (i >= quint64(countA) || j >= quint64(countB))
        fail("Correspondence index exceeds feature count");
      const auto index = key(quint32(i), quint32(j));
      if (verified) {
        if (verifiedIndices.contains(index) || empty)
          fail("Duplicate/inconsistent inlier");
        verifiedIndices.insert(index);
        ++seen;
      }
      auto& m = matches[index];
      m.a = quint32(i);
      m.b = quint32(j);
      m.inlier |= verified != 0;
    }
  }
  if (result.verified && (seen != declared || (declared == 0 && !empty)))
    fail("Verified inlier count mismatch");
  for (auto m : matches) {
    m.verified = result.verified;
    result.inliers += m.inlier ? 1 : 0;
    result.matches.push_back(m);
  }
  return result;
}

void validateTexture(const QString& path, const std::function<bool()>& cancelled);
QImage readTexturePreview(const QString& path, const std::function<bool()>& cancelled);
static SceneData readSceneImpl(const QString& path, const SceneProgress& progress,
                   const std::function<bool()>& cancelled, bool validation) {
  auto report = [&](const QString& stage, int percent = -1) {
    if (cancelled && cancelled()) fail("Scene loading cancelled");
    if (progress) progress(stage, percent);
  };
  report(QString::fromUtf8("读取模型"));
  SceneData data;
  data.source = path;
  if (path.isEmpty()) return data;
  const bool directory = QFileInfo(path).isDir();
  auto complete = [&](const QString& extension) {
    for (const auto& name : {"cameras", "images", "points3D"})
      if (!QFileInfo::exists(QDir(path).filePath(QString(name) + extension)))
        return false;
    return true;
  };
  const bool binary = directory && complete(".bin");
  if (directory && !binary && !complete(".txt"))
    fail("Incomplete COLMAP model (cameras/images/points3D required)");
  QFile file(directory
                 ? QDir(path).filePath(binary ? "points3D.bin" : "points3D.txt")
                 : path);
  open(file);
  if (directory && file.fileName().endsWith(".bin")) {
    QDataStream s(&file);
    s.setByteOrder(QDataStream::LittleEndian);
    s.setFloatingPointPrecision(QDataStream::DoublePrecision);
    quint64 count = 0;
    s >> count;
    if (count > 100000000 || count > quint64(file.size() / 43))
      fail("Invalid sparse point count");
    data.totalPoints = count;
    const auto stride = qMax<quint64>(1, (count + 1999999) / 2000000);
    for (quint64 i = 0; i < count; ++i) {
      if (i % 8192 == 0) report(QString::fromUtf8("读取稀疏点"), int(100 * i / qMax<quint64>(1, count)));
      quint64 id = 0, tracks = 0;
      double x = 0, y = 0, z = 0, error = 0;
      quint8 r = 0, g = 0, b = 0;
      s >> id >> x >> y >> z >> r >> g >> b >> error >> tracks;
      streamOk(s);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
          tracks > quint64((file.size() - file.pos()) / 8))
        fail("Invalid sparse point");
      if (!validation && i % stride == 0)
        data.points.push_back(
            {float(x), float(y), float(z), r / 255.f, g / 255.f, b / 255.f});
      if (!file.seek(file.pos() + qint64(tracks) * 8))
        fail("Truncated sparse track");
    }
    streamOk(s);
  } else if (directory) {
    QTextStream text(&file);
    while (!text.atEnd()) {
      if (data.totalPoints % 8192 == 0) report(QString::fromUtf8("读取稀疏点"));
      const auto line = text.readLine();
      if (line.isEmpty() || line.startsWith('#')) continue;
      const auto row = fields(line);
      if (row.size() < 8) fail("Malformed points3D.txt");
      ++data.totalPoints;
      const SceneVertex point{float(real(row[1])), float(real(row[2])),
                               float(real(row[3])), float(real(row[4]) / 255),
                               float(real(row[5]) / 255),
                               float(real(row[6]) / 255)};
      if (!validation && data.points.size() < 2000000) data.points.push_back(point);
    }
  } else {
    // PLY scalar properties are read by declared names, not assumed column
    // offsets.
    QString format;
    QStringList properties;
    QVector<QStringList> faceProperties;
    quint64 vertices = 0, faces = 0;
    QString element;
    bool ended = false;
    auto headerLine = [&]() {
      const auto line = QString::fromLatin1(file.readLine(16384)).trimmed();
      if (line.size() >= 16383) fail("PLY header line too long");
      return line;
    };
    if (headerLine() != "ply") fail("Expected PLY file");
    for (int n = 0; n < 10000 && !file.atEnd(); ++n) {
      const auto line = headerLine();
      const auto row = fields(line);
      if (row.isEmpty()) continue;
      if (row[0] == "end_header") {
        ended = true;
        break;
      }
      if (row[0] == "format" && row.size() >= 2) format = row[1];
      if (row[0] == "element" && row.size() == 3) {
        element = row[1];
        if (element == "vertex")
          vertices = number(row[2]);
        else if (element == "face")
          faces = number(row[2]);
        else if (number(row[2]) != 0)
          fail("Unsupported nonempty PLY element");
      }
      if (row[0] == "property") {
        if (element == "vertex") {
          if (row.size() != 3) fail("Unsupported PLY vertex list");
          properties << row[1] + ":" + row[2];
        } else if (element == "face")
          faceProperties.push_back(row.mid(1));
      }
      if (line.startsWith("comment TextureFile "))
        data.texture = QFileInfo(path).dir().filePath(line.mid(20));
    }
    if (!ended || !(format == "ascii" || format == "binary_little_endian"))
      fail("Unsupported PLY format");
    const auto fileSize = file.size();
    if (vertices > quint64(fileSize) || faces > quint64(fileSize))
      fail("PLY counts exceed file length");
    constexpr quint64 kMeshBudget = 1024ULL * 1024 * 1024;
    constexpr quint64 kMaxPreviewTriangles = 500000;
    constexpr quint64 kMaxPoints = 2000000;
    const auto pointBudget = qMin(vertices, kMaxPoints);
    const bool sample = !validation &&
        (vertices > kMeshBudget / sizeof(SceneVertex) || faces > kMeshBudget / (3 * sizeof(SceneVertex)) ||
         (vertices + pointBudget + faces * 3) * sizeof(SceneVertex) > kMeshBudget);
    data.sampledMesh = sample && faces > 0;
    const bool pointCloud = !validation && faces == 0;
    const bool ascii = format == "ascii";
    const auto vertexStart = file.pos();
    QDataStream s(&file);
    s.setByteOrder(QDataStream::LittleEndian);
    auto scalar = [&](const QString& type) -> double {
      if (type == "float" || type == "float32") {
        float v;
        s.setFloatingPointPrecision(QDataStream::SinglePrecision);
        s >> v;
        return v;
      }
      if (type == "double" || type == "float64") {
        double v;
        s.setFloatingPointPrecision(QDataStream::DoublePrecision);
        s >> v;
        return v;
      }
      if (type == "uchar" || type == "uint8") {
        quint8 v;
        s >> v;
        return v;
      }
      if (type == "char" || type == "int8") {
        qint8 v;
        s >> v;
        return v;
      }
      if (type == "uint" || type == "uint32") {
        quint32 v;
        s >> v;
        return v;
      }
      if (type == "int" || type == "int32") {
        qint32 v;
        s >> v;
        return v;
      }
      if (type == "ushort" || type == "uint16") {
        quint16 v;
        s >> v;
        return v;
      }
      if (type == "short" || type == "int16") {
        qint16 v;
        s >> v;
        return v;
      }
      fail("Unsupported PLY scalar type");
    };
    QVector<SceneVertex> all;
    if (!validation && !sample && !pointCloud) all.reserve(int(vertices));
    if (!validation && !sample && faces) data.triangles.reserve(int(faces * 3));
    QStringList types, names;
    for (const auto& property : properties) {
      const auto prop = property.split(':');
      types << prop[0]; names << prop[1];
    }
    const int x = names.indexOf("x"), y = names.indexOf("y"), z = names.indexOf("z");
    if (x < 0 || y < 0 || z < 0) fail("PLY lacks XYZ");
    const int red = names.indexOf("red"), green = names.indexOf("green"), blue = names.indexOf("blue");
    const int u = names.contains("u") ? names.indexOf("u") : names.indexOf("s");
    const int v = names.contains("v") ? names.indexOf("v") : names.indexOf("t");
    auto scalarBytes = [](const QString& type) {
      if (type == "uchar" || type == "uint8" || type == "char" || type == "int8") return 1;
      if (type == "ushort" || type == "uint16" || type == "short" || type == "int16") return 2;
      if (type == "float" || type == "float32" || type == "uint" || type == "uint32" ||
          type == "int" || type == "int32") return 4;
      if (type == "double" || type == "float64") return 8;
      fail("Unsupported PLY scalar type");
    };
    auto skipBinary = [&](qint64 bytes) {
      if (bytes > fileSize - file.pos()) fail("Truncated PLY data");
      if (bytes <= 65536) {
        if (s.skipRawData(int(bytes)) != bytes) fail("Truncated PLY data");
      } else if (!file.seek(file.pos() + bytes)) fail("Truncated PLY data");
    };
    auto readAsciiLine = [&] {
      const auto line = file.readLine(65536);
      if (line.isEmpty() || line.size() == 65535) fail("Missing or oversized PLY record");
      return line;
    };
    enum class FaceField { Scalar, IndexList, Texcoords, OtherList };
    struct FaceProperty { FaceField field; QString type, valueType; int itemBytes; };
    QVector<FaceProperty> faceLayout;
    for (const auto& prop : faceProperties) {
      if (prop.size() == 2) faceLayout.push_back({FaceField::Scalar, prop[0], {}, 0});
      else if (prop.size() == 4 && prop[0] == "list") {
        const bool indices = prop[3] == "vertex_indices" || prop[3] == "vertex_index";
        faceLayout.push_back({indices ? FaceField::IndexList : prop[3] == "texcoord" ? FaceField::Texcoords : FaceField::OtherList,
                              prop[1], prop[2], ascii ? 0 : scalarBytes(prop[2])});
      } else fail("Malformed PLY face property");
    }
    int vertexBytes = 0;
    if (!ascii) for (const auto& type : types) vertexBytes += scalarBytes(type);
    QVector<double> values(properties.size());
    auto readVertex = [&] {
      QStringList row;
      if (ascii) row = fields(QString::fromLatin1(readAsciiLine()));
      if (ascii && row.size() != properties.size())
        fail("Malformed PLY vertex");
      for (int j = 0; j < properties.size(); ++j) {
        values[j] = ascii ? real(row[j]) : scalar(types[j]);
      }
      SceneVertex p{float(values[x]), float(values[y]), float(values[z]),
                    float((red < 0 ? 160 : values[red]) / 255),
                    float((green < 0 ? 160 : values[green]) / 255),
                    float((blue < 0 ? 160 : values[blue]) / 255)};
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        fail("Non-finite PLY coordinate");
      p.u = u < 0 ? 0 : float(values[u]);
      p.v = v < 0 ? 0 : float(values[v]);
      return p;
    };
    auto skipVertices = [&](quint64 count) {
      if (!count) return;
      if (!ascii) {
        if (!vertexBytes || count > quint64((fileSize - file.pos()) / vertexBytes))
          fail("Truncated PLY vertex data");
        skipBinary(qint64(count) * vertexBytes);
        return;
      }
      for (quint64 i = 0; i < count; ++i) {
        if (i % 8192 == 0 && cancelled && cancelled()) fail("Scene loading cancelled");
        readAsciiLine();
      }
    };
    if (data.sampledMesh) skipVertices(vertices);
    else if (pointCloud && vertices > 0) {
      const auto stride = 1 + (vertices - 1) / kMaxPoints;
      data.points.reserve(int(1 + (vertices - 1) / stride));
      for (quint64 i = 0; i < vertices;) {
        if (data.points.size() % 8192 == 0)
          report(QString::fromUtf8("读取顶点"), int(100 * i / vertices));
        data.points << readVertex();
        const auto count = qMin(stride, vertices - i);
        skipVertices(count - 1);
        i += count;
      }
    } else {
      for (quint64 i = 0; i < vertices; ++i) {
        if (i % 8192 == 0) report(QString::fromUtf8("读取顶点"), int(100 * i / qMax<quint64>(1, vertices)));
        const auto p = readVertex();
        if (!validation) all << p;
      }
    }
    struct PreviewCorner { quint64 index; float u, v; bool hasUv; };
    QVector<PreviewCorner> previewCorners;
    const auto faceStride = faces ? 1 + (faces - 1) / kMaxPreviewTriangles : 1;
    for (quint64 i = 0; i < faces; ++i) {
      if (i % 4096 == 0) report(QString::fromUtf8("读取网格"), int(100 * i / qMax<quint64>(1, faces)));
      const bool keep = !sample || (i % faceStride == 0 && quint64(previewCorners.size()) < kMaxPreviewTriangles * 3);
      QStringList row;
      int cursor = 0;
      if (ascii) {
        const auto line = readAsciiLine();
        if (!keep && !faceLayout.isEmpty() && faceLayout.first().field == FaceField::IndexList) {
          const auto end = line.constEnd();
          auto begin = line.constBegin();
          while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
          int count = 0;
          const auto parsed = std::from_chars(begin, end, count);
          if (parsed.ec != std::errc{} || count < 3 || count > 1024 ||
              (parsed.ptr != end && !std::isspace(static_cast<unsigned char>(*parsed.ptr))))
            fail("Invalid PLY list count");
          data.totalTriangles += count - 2;
          continue;
        }
        row = fields(QString::fromLatin1(line));
      }
      auto next = [&](const QString& type) {
        if (!ascii) return scalar(type);
        if (cursor >= row.size()) fail("Truncated PLY face");
        return real(row[cursor++]);
      };
      QVector<quint64> indices;
      QVector<float> uv;
      int indexCount = 0;
      for (const auto& prop : faceLayout) {
        if (prop.field != FaceField::Scalar) {
          double raw = next(prop.type);
          if (raw < 0 || raw > 1024 || raw != std::floor(raw))
            fail("Invalid PLY list count");
          const int count = int(raw);
          if (prop.field == FaceField::IndexList) indexCount += count;
          if (!keep && !ascii) {
            const int bytes = count * prop.itemBytes;
            if (s.skipRawData(bytes) != bytes) fail("Truncated PLY face");
            continue;
          }
          for (int k = 0; k < count; ++k) {
            double value = next(prop.valueType);
            if (prop.field == FaceField::IndexList) {
              if (!std::isfinite(value) || value < 0 || value >= double(vertices) ||
                  value != std::floor(value))
                fail("Invalid PLY face index");
              indices << quint64(value);
            } else if (prop.field == FaceField::Texcoords)
              uv << float(value);
          }
        } else next(prop.type);
      }
      if (indexCount < 3) fail("PLY face has fewer than 3 vertices");
      if (ascii && cursor != row.size()) fail("Unexpected PLY face fields");
      data.totalTriangles += quint64(indexCount - 2);
      if (validation || !keep) continue;
      if (!sample && (vertices + pointBudget + 3 * data.totalTriangles) * sizeof(SceneVertex) > kMeshBudget)
        fail("Triangulated PLY exceeds the 1 GiB display budget");
      const bool hasUv = uv.size() == indices.size() * 2;
      for (int k = 1; k < indices.size() - 1; ++k) {
        if (sample && quint64(previewCorners.size()) >= kMaxPreviewTriangles * 3) break;
        for (int corner : {0, k, k + 1}) {
          if (sample) {
            previewCorners.push_back({indices[corner], hasUv ? uv[corner * 2] : 0.f,
                                      hasUv ? uv[corner * 2 + 1] : 0.f, hasUv});
            continue;
          }
          auto p = all[int(indices[corner])];
          if (hasUv) {
            p.u = uv[corner * 2];
            p.v = uv[corner * 2 + 1];
          }
          data.triangles << p;
        }
      }
    }
    if (!ascii) streamOk(s);
    if (data.sampledMesh) {
      // Keep only the vertices referenced by sampled faces. The source mesh
      // and face UVs remain unchanged, and the preview can stay within budget.
      QVector<quint64> ids;
      ids.reserve(previewCorners.size());
      for (const auto& corner : previewCorners) ids << corner.index;
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      if (!file.seek(vertexStart)) fail("Cannot reopen PLY vertices");
      quint64 previous = 0;
      all.reserve(ids.size());
      for (int i = 0; i < ids.size(); ++i) {
        if (i % 8192 == 0) report(QString::fromUtf8("准备网格预览"), 100 * i / ids.size());
        skipVertices(ids[i] - previous);
        all << readVertex();
        previous = ids[i] + 1;
      }
      if (!ascii) streamOk(s);
      data.triangles.reserve(previewCorners.size());
      for (int i = 0; i < previewCorners.size(); ++i) {
        if (i % 8192 == 0) report(QString::fromUtf8("组装网格预览"), 100 * i / previewCorners.size());
        const auto& corner = previewCorners[i];
        auto p = all[int(std::lower_bound(ids.cbegin(), ids.cend(), corner.index) - ids.cbegin())];
        if (corner.hasUv) { p.u = corner.u; p.v = corner.v; }
        data.triangles << p;
      }
    }
    data.totalPoints = vertices;
    const auto stride = qMax(1, (all.size() + int(kMaxPoints) - 1) / int(kMaxPoints));
    for (int i = 0; i < all.size(); i += stride) data.points << all[i];
  }
  if (directory) {
    report(QString::fromUtf8("读取相机"));
    QVector3D low(1e30f, 1e30f, 1e30f), high(-1e30f, -1e30f, -1e30f);
    for (const auto& p : data.points) {
      low.setX(qMin(low.x(), p.x));
      low.setY(qMin(low.y(), p.y));
      low.setZ(qMin(low.z(), p.z));
      high.setX(qMax(high.x(), p.x));
      high.setY(qMax(high.y(), p.y));
      high.setZ(qMax(high.z(), p.z));
    }
    const float size = data.points.isEmpty()
                           ? .1f
                           : qMax(.001f, (high - low).length() * .025f);
    auto camera = [&](double qw, double qx, double qy, double qz, double tx,
                      double ty, double tz) {
      QQuaternion q{float(qw), float(qx), float(qy), float(qz)};
      if (!std::isfinite(q.length()) || q.length() < 1e-8 ||
          !std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz))
        fail("Invalid camera pose");
      const auto inverse = q.normalized().conjugated();
      if (validation) return;
      const auto center =
          inverse.rotatedVector({float(-tx), float(-ty), float(-tz)});
      QVector3D corners[4];
      for (int i = 0; i < 4; ++i)
        corners[i] = center + inverse.rotatedVector(
                                  {(i == 0 || i == 3 ? -1 : 1) * size,
                                   (i < 2 ? -1 : 1) * size * .7f, size * 1.5f});
      auto line = [&](const QVector3D& p) {
        data.cameras.push_back({p.x(), p.y(), p.z(), .72f, .32f, .25f});
      };
      for (int i = 0; i < 4; ++i) {
        line(center);
        line(corners[i]);
        line(corners[i]);
        line(corners[(i + 1) % 4]);
      }
    };
    QFile images(QDir(path).filePath(binary ? "images.bin" : "images.txt"));
    open(images);
    if (binary) {
      QDataStream s(&images);
      s.setByteOrder(QDataStream::LittleEndian);
      s.setFloatingPointPrecision(QDataStream::DoublePrecision);
      quint64 count = 0;
      s >> count;
      if (count > 1000000) fail("Excessive camera count");
      for (quint64 i = 0; i < count; ++i) {
        if (i % 128 == 0) report(QString::fromUtf8("读取相机"), int(100 * i / qMax<quint64>(1, count)));
        quint32 id = 0, calibration = 0;
        double qw, qx, qy, qz, tx, ty, tz;
        s >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> calibration;
        streamOk(s);
        char c = 1;
        int length = 0;
        while (c != 0) {
          if (++length > 32768 || !images.getChar(&c))
            fail("Invalid image name");
        }
        quint64 points = 0;
        s >> points;
        streamOk(s);
        if (points > quint64((images.size() - images.pos()) / 24) ||
            !images.seek(images.pos() + qint64(points) * 24))
          fail("Invalid image observations");
        camera(qw, qx, qy, qz, tx, ty, tz);
      }
    } else {
      QTextStream text(&images);
      while (!text.atEnd()) {
        const auto line = text.readLine();
        if (line.isEmpty() || line.startsWith('#')) continue;
        const auto row = fields(line);
        if (row.size() < 10) fail("Invalid images.txt pose");
        camera(real(row[1]), real(row[2]), real(row[3]), real(row[4]),
               real(row[5]), real(row[6]), real(row[7]));
        if (text.atEnd()) fail("Missing image observation row");
        text.readLine();
      }
    }
  }
  if (!data.texture.isEmpty()) {
    report(QString::fromUtf8("解码纹理"));
    if (validation) {
      validateTexture(data.texture, cancelled);
      return data;
    }
    data.textureRevision = fileRevision(data.texture);
    data.texturePreview = readTexturePreview(data.texture, cancelled);
  }
  report(QString::fromUtf8("模型读取完成"), 100);
  return data;
}
SceneData readScene(const QString& path, const SceneProgress& progress,
                   const std::function<bool()>& cancelled) {
  return readSceneImpl(path, progress, cancelled, false);
}
ArtifactSummary validateScene(const QString& path, const std::function<bool()>& cancelled) {
  const auto data = readSceneImpl(path, {}, cancelled, true);
  return {data.totalPoints, data.totalTriangles, data.texture};
}
}  // namespace colite::gui
