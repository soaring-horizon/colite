cmake_minimum_required(VERSION 3.20)
include(cmake/ColiteVersion.cmake)

project(
  colite
  VERSION ${COLITE_VERSION}
  DESCRIPTION "A compact C++ SfM and MVS reconstruction system"
  LANGUAGES C CXX
)

include(GNUInstallDirs)
if(MSVC)
  add_compile_definitions(_USE_MATH_DEFINES)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR})
  include(InstallRequiredSystemLibraries)
endif()

option(COLITE_BUILD_TESTS "Build unit tests" ON)
option(COLITE_BUILD_GUI "Build the optional Qt 5.15.2 desktop workbench" OFF)
option(COLITE_ENABLE_CUDA_SIFT "Build the optimized CUDA SIFT feature backend when CUDA is available" ON)
option(COLITE_ENABLE_CUDA_MATCHER "Build the CUDA brute-force descriptor matcher" ON)
option(COLITE_ENABLE_CUDA_MVS "Build the optional CUDA MVS PatchMatch backend" OFF)
option(COLITE_ENABLE_SIFT_PARALLEL "Enable the persistent SIFT worker pool" ON)
option(COLITE_ENABLE_AVX2 "Enable AVX2/FMA optimized CPU SIFT kernels" ON)
option(COLITE_ENABLE_POISSON_MESHING "Build the bundled PoissonRecon mesher" ON)
option(COLITE_ENABLE_DELAUNAY_MESHING "Build the CGAL Delaunay mesher when CGAL is available" ON)
option(COLITE_ENABLE_ONNX "Build optional ONNX Runtime learned feature backends" OFF)

if(COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS)
  # WSL non-login shells commonly omit /usr/local/cuda/bin from PATH even
  # when the toolkit is installed there.
  if(UNIX AND NOT CMAKE_CUDA_COMPILER AND
     EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
        "CUDA compiler" FORCE)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
endif()

if((COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS) AND
   CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
elseif(COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS)
  message(STATUS "CUDA compiler not found: CUDA SIFT/matcher/MVS backends are disabled")
  set(COLITE_ENABLE_CUDA_SIFT OFF)
  set(COLITE_ENABLE_CUDA_MATCHER OFF)
  set(COLITE_ENABLE_CUDA_MVS OFF)
endif()

install(
  FILES
    LICENSE
    THIRD_PARTY_NOTICES.md
    README.md
    README.en.md
    INSTALL.md
    CHANGELOG.md
    CONTRIBUTING.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite
)

install(
  FILES
    thirdparty/PoissonRecon/LICENSE
    thirdparty/PoissonRecon/NOTICE.txt
    thirdparty/PoissonRecon/README.colite.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite/licenses/PoissonRecon
)

install(
  DIRECTORY docs/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite/docs
  FILES_MATCHING PATTERN "*.md" PATTERN "*.png"
)

# Keep both offline bridges and their local Python imports together. These
# tools exchange files; the reconstruction backend does not use SQLite.
install(
  FILES
    scripts/convert_colmap_database_to_colite_workspace.py
    scripts/convert_colite_workspace_to_colmap_database.py
    scripts/colite_workspace_features.py
    scripts/colite_workspace_geometry.py
    scripts/analyze_sfm_geometric_merges.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite/scripts
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(PoseLib QUIET)
find_library(FLANN_CPP_LIBRARY NAMES flann_cpp REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

if(COLITE_ENABLE_ONNX)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an unpacked ONNX Runtime SDK")
  if(NOT ONNXRUNTIME_ROOT)
    find_package(onnxruntime CONFIG QUIET)
  endif()
  if(NOT TARGET onnxruntime::onnxruntime)
    # Official Windows SDK archives ship headers/import libraries without a
    # package config. Accept the same SDK root on either platform.
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
      HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
      HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
  endif()
endif()

if(COLITE_ENABLE_DELAUNAY_MESHING)
  find_package(CGAL QUIET)
  if(NOT CGAL_FOUND)
    message(STATUS "CGAL not found: Delaunay meshing will be unavailable")
  endif()
endif()

if(COLITE_ENABLE_POISSON_MESHING)
  add_subdirectory(thirdparty/PoissonRecon)
endif()

add_subdirectory(src/colite)
add_subdirectory(src/colite/exe)

if(COLITE_BUILD_GUI)
  add_subdirectory(gui)
endif()

if(COLITE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests)
endif()
