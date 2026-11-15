# Third-party notices

Colite is distributed under the BSD 3-Clause License. Some source files are
adapted from, or designed for compatibility with, the projects listed below.
Their licenses remain applicable to those portions. Dependencies installed by
the user retain their own licenses. Desktop binary packages also carry runtime
dependencies; their notices and source inventory are included under
`share/colite/licenses/runtime` and `share/colite/sources` in each package.

## Desktop workbench

The workbench dynamically links Qt 5.15.2. Its Qt Base source archive, license
texts and third-party attribution files accompany desktop packages. The source
archive and SDK version are recorded in the package evidence. Qt runtime
libraries can be replaced independently of Colite.

The bundled Phosphor icons use the MIT License; the notice is retained at
`gui/assets/PHOSPHOR-LICENSE` and installed under
`share/colite/licenses/Phosphor`.

## COLMAP

Homography decomposition in `src/colite/matching/homography_pose.cpp` is adapted
from COLMAP 4.0 `geometry/homography_matrix.cc` and retains its BSD notice.

The EPnP solver in `thirdparty/colmap/epnp.{h,cpp}` is adapted from COLMAP 4.0
`estimators/solvers/absolute_pose.{h,cc}`. Its BSD notice, source checksums and
local changes are retained alongside the adapter.

Portions of the meshing and reconstruction compatibility code are adapted
from, or based on behavior and data formats from, COLMAP.

Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the names of ETH Zurich and UNC Chapel Hill nor the names of their
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## PoseLib

The builtin P3P solver in `thirdparty/poselib` is adapted from PoseLib v2.0.4.
Only the minimal solver, depth refinement, and required pose conversions are
bundled; provenance and local changes are recorded in that directory.

The adapted quaternion conversions retain Copyright (c) 2021, Viktor Larsson,
under the same BSD 3-Clause terms.

BSD 3-Clause License

Copyright (c) 2020, Viktor Larsson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## VLFeat

`src/colite/features/backends/sift_vlfeat.c` contains an optimized adaptation
of VLFeat SIFT routines.

Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson
Copyright (C) 2012-13, The VLFeat Team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## PoissonRecon

PoissonRecon is bundled under `thirdparty/PoissonRecon` and is licensed under
the MIT License. Its complete license is retained in
`thirdparty/PoissonRecon/LICENSE`. Individual source files additionally retain
BSD copyright and license notices in their headers; these notices are preserved
alongside the directory license. `thirdparty/PoissonRecon/NOTICE.txt` collects
the distinct original source headers and the directory license. Installed
packages include these files under `share/colite/licenses/PoissonRecon`.

Copyright (c) 2015 mkazhdan

## Other build dependencies

Colite can link against Eigen, OpenCV, Ceres Solver, glog, FLANN, LZ4, CGAL,
Boost, CUDA and optionally PoseLib. These packages are discovered from the
build environment and are not vendored by Colite. See each upstream project
for its license and redistribution requirements.

`references/`, local datasets and OpenMVS/vcglib reference builds are not part
of the public source distribution and are not linked into Colite.
