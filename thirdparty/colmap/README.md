# COLMAP EPnP adapter

Source: COLMAP 4.0.0, `src/colmap/estimators/solvers/absolute_pose.{h,cc}`.
License: BSD 3-Clause, retained in both source headers and LICENSE.
Only EPnP is included. No COLMAP runtime, database, P3P/P4PF or RANSAC dependency.

Local changes: isolate namespace/types/includes, omit unused Residuals interface,
replace logging preconditions with failure returns, skip non-finite pose candidates.
All projection and inverse projection remain callbacks into Colite camera models.
The algebraic EPnP/control-point/beta/Gauss-Newton calculations are unchanged.

Original upstream file SHA256:
- `absolute_pose.h`: `6b11533e550c3197cf038af366fe6e7759b4dc694546835fb7286074236a4777`
- `absolute_pose.cc`: `613e83713bf23a82d4b62e40bcd72be9aa2c10e724633a807b8c320123b19e38`
