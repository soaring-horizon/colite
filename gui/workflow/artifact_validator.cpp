#include <QFile>
#include <QImage>
#include <QImageIOHandler>
#include <QImageReader>
#include <png.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace colite::gui {
namespace {
struct PngReader {
  QFile file;
  png_structp png = nullptr;
  png_infop info = nullptr;
  png_bytep row = nullptr;
  std::function<bool()> cancelled;
  QSize previewSize;
  QImage preview;
  ~PngReader() {
    std::free(row);
    if (png) png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
  }
};
// Keep libpng's longjmp boundary free of C++ automatic objects with destructors.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4611) // All owning objects live outside this C API boundary.
#endif
bool decodePng(PngReader* reader) {
  if (setjmp(png_jmpbuf(reader->png))) return false;
  png_set_read_fn(reader->png, &reader->file, [](png_structp png, png_bytep out, png_size_t size) {
    auto* file = static_cast<QFile*>(png_get_io_ptr(png));
    if (file->read(reinterpret_cast<char*>(out), qint64(size)) != qint64(size))
      png_error(png, "Truncated texture");
  });
  png_set_crc_action(reader->png, PNG_CRC_ERROR_QUIT, PNG_CRC_ERROR_QUIT);
  png_read_info(reader->png, reader->info);
  const auto width = png_get_image_width(reader->png, reader->info);
  const auto height = png_get_image_height(reader->png, reader->info);
  if (reader->previewSize.isValid()) {
    if (reader->previewSize.width() > int(width) || reader->previewSize.height() > int(height))
      return false;
    const auto colorType = png_get_color_type(reader->png, reader->info);
    const bool hasAlpha = (colorType & PNG_COLOR_MASK_ALPHA) ||
                          png_get_valid(reader->png, reader->info, PNG_INFO_tRNS);
    png_set_expand(reader->png);
    png_set_strip_16(reader->png);
    if ((colorType & PNG_COLOR_MASK_COLOR) == 0) png_set_gray_to_rgb(reader->png);
    if (!hasAlpha) png_set_add_alpha(reader->png, 255, PNG_FILLER_AFTER);
    reader->preview = QImage(reader->previewSize, QImage::Format_RGBA8888);
    if (reader->preview.isNull()) return false;
    reader->preview.fill(Qt::transparent);
  }
  const int passes = png_set_interlace_handling(reader->png);
  png_read_update_info(reader->png, reader->info);
  reader->row = static_cast<png_bytep>(std::calloc(png_get_rowbytes(reader->png, reader->info), 1));
  if (!reader->row) return false;
  for (int pass = 0; pass < passes; ++pass) {
    int outputY = 0;
    for (png_uint_32 y = 0; y < height; ++y) {
      if (y % 256 == 0 && reader->cancelled && reader->cancelled()) return false;
      png_read_row(reader->png, reader->row, nullptr);
      if (reader->preview.isNull() || outputY >= reader->preview.height() ||
          y != quint64(outputY) * height / reader->preview.height()) continue;
      auto* dst = reader->preview.scanLine(reader->preview.height() - 1 - outputY++);
      if (passes == 7 && !PNG_ROW_IN_INTERLACE_PASS(y, pass)) continue;
      for (int x = 0; x < reader->preview.width(); ++x) {
        const auto sourceX = quint64(x) * width / reader->preview.width();
        if (passes != 7 || PNG_COL_IN_INTERLACE_PASS(sourceX, pass))
          std::memcpy(dst + x * 4, reader->row + sourceX * 4, 4);
      }
    }
  }
  png_read_end(reader->png, reader->info);
  return true;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
void readPng(PngReader& reader) {
  reader.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!reader.png) throw std::runtime_error("Cannot allocate PNG decoder");
  reader.info = png_create_info_struct(reader.png);
  if (!reader.info) throw std::runtime_error("Cannot allocate PNG metadata");
  if (!decodePng(&reader)) throw std::runtime_error("Invalid or truncated PNG texture");
}
}
void validateTexture(const QString& path, const std::function<bool()>& cancelled) {
  // Generated atlases are PNG. Decode one scanline at a time, including every
  // interlace pass and CRC, rather than allocating the full-resolution atlas.
  PngReader reader;
  reader.cancelled = cancelled;
  reader.file.setFileName(path);
  if (!reader.file.open(QIODevice::ReadOnly))
    throw std::runtime_error("Cannot open texture");
  if (reader.file.peek(8) != QByteArray::fromHex("89504e470d0a1a0a")) {
    QImageReader image(path);
    const auto size = image.size();
    if (!size.isValid()) throw std::runtime_error("Invalid texture image");
    image.setScaledSize(size.scaled(1024, 1024, Qt::KeepAspectRatio));
    if (image.read().isNull()) throw std::runtime_error("Cannot decode texture image");
    return;
  }
  readPng(reader);
}

QImage readTexturePreview(const QString& path, const std::function<bool()>& cancelled) {
  QImageReader image(path);
  const auto size = image.size();
  if (!size.isValid() || size.isEmpty()) throw std::runtime_error("Invalid texture image");
  const auto target = size.boundedTo(size.scaled(4096, 4096, Qt::KeepAspectRatio).expandedTo(QSize(1, 1)));
  constexpr qint64 kMaxDirectDecodePixels = 128LL * 1024 * 1024;
  if (qint64(size.width()) * size.height() > kMaxDirectDecodePixels) {
    if (image.format() == "png") {
      PngReader reader;
      reader.cancelled = cancelled;
      reader.previewSize = target;
      reader.file.setFileName(path);
      if (!reader.file.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot open texture");
      readPng(reader);
      return reader.preview;
    }
    if (!image.supportsOption(QImageIOHandler::ScaledSize)) return {};
  }
  image.setScaledSize(target);
  const auto preview = image.read();
  if (preview.isNull()) throw std::runtime_error("Cannot decode texture image");
  return preview.scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation).mirrored();
}
}
