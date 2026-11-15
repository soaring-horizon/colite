# PoseLib P3P adapter

Source: PoseLib v2.0.4, `PoseLib/solvers/p3p.{h,cc}`, `p3p_common.h`, and
the minimal matrix/quaternion conversions from `camera_pose.h` and
`misc/quaternion.h`. BSD 3-Clause; see LICENSE and retained source notices.
The pinned upstream URLs and original SHA256 values are in `upstream.json`.

Only the calibrated three-point solver and its Newton depth refinement are
bundled. No robust estimator, camera models, generalized solver, runtime
PoseLib library, or new external dependency is included.

Local changes: isolate the namespace, reduce CameraPose to the required
quaternion/translation conversions, reject invalid/non-unit rays and
duplicate/collinear point samples before inversion, discard non-finite roots.
The nondegenerate P3P algebra and root ordering are unchanged. The builtin
implementation replaces six-point DLT when the optional PoseLib dependency
is absent; builds with PoseLib keep using the installed P3P solver.
