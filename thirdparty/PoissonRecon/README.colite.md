# PoissonRecon vendoring note

This directory contains the PoissonRecon source bundled with COLMAP 3.12.6.
COLITE builds these sources directly as the `colite_poisson_recon` static
library; no system PoissonRecon package, download, or separate installation is
required.

Upstream project: PoissonRecon by Michael Kazhdan and Matthew Bolitho.

The directory `LICENSE` contains the upstream MIT license. Individual source
files also retain BSD copyright and license notices; preserve those headers.
`NOTICE.txt` collects the distinct original source headers and the directory
license for installed distributions. Update it when changing the bundled source
version or its notices.

Colite integration changes expose a scoped thread-pool budget and clear command
flags between embedded calls. The wrapper explicitly sets fullDepth for every
invocation and serializes access to the upstream process-wide parser state.

The integration wrapper lives in `src/colite/mvs/meshing.cpp`. COLMAP is the
behavioral reference for the command-line arguments passed to `PoissonRecon`
and `SurfaceTrimmer`.
