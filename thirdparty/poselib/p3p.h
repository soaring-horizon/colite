// Copyright (c) 2020, Viktor Larsson
// Copyright (c) 2021, Viktor Larsson (quaternion conversion).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Adapted from PoseLib v2.0.4; isolated P3P and a minimal pose type only.
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace colite::third_party::poselib {
struct CameraPose {
    Eigen::Vector4d q;
    Eigen::Vector3d t;
    CameraPose() : q(1, 0, 0, 0), t(0, 0, 0) {}
    CameraPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& translation) : t(translation) {
        const Eigen::Quaterniond quat(R);
        q << quat.w(), quat.x(), quat.y(), quat.z();
        q.normalize();
    }
    Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> result;
        result.leftCols<3>() = Eigen::Quaterniond(q[0], q[1], q[2], q[3]).toRotationMatrix();
        result.col(3) = t;
        return result;
    }
};
// Three finite, unit bearing vectors and non-collinear world points.
// Returns every finite camera-from-world candidate with positive sample depths.
int p3p(const std::vector<Eigen::Vector3d>& x, const std::vector<Eigen::Vector3d>& X,
        std::vector<CameraPose>* output);
}  // namespace colite::third_party::poselib
