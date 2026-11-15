param(
  [string]$QtRoot = $env:COLITE_QT_ROOT,
  [string]$VcpkgRoot = $env:VCPKG_ROOT,
  [string]$BuildRoot = '',
  [string]$DependencyRoot = '',
  [string]$InstallRoot = '',
  [string]$Generator = 'Visual Studio 16 2019',
  [ValidateSet('x64-windows-colite', 'x64-windows-colite-v143')]
  [string]$Triplet = 'x64-windows-colite',
  [string]$PythonExecutable = '',
  [switch]$Cuda,
  [string]$CudaToolkitRoot = $env:CUDA_PATH,
  [string]$CudaArchitectures = '',
  [int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (!$QtRoot -or !$VcpkgRoot) { throw 'Set COLITE_QT_ROOT (Qt 5.15.2 MSVC x64) and VCPKG_ROOT.' }
$QtRoot = (Resolve-Path -LiteralPath $QtRoot).Path
$VcpkgRoot = (Resolve-Path -LiteralPath $VcpkgRoot).Path
if (!$BuildRoot) { $BuildRoot = Join-Path $repo 'output/windows-release/build' }
if (!$DependencyRoot) { $DependencyRoot = Join-Path $repo 'output/windows-release/dependencies' }
if (!$InstallRoot) { $InstallRoot = Join-Path $repo 'output/windows-release/install' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$support = Join-Path (Split-Path $DependencyRoot) 'vcpkg-support'
New-Item -ItemType Directory -Force $support | Out-Null
$downloads = Join-Path $support 'downloads'
$triplets = Join-Path $repo 'cmake/triplets'
$vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
if (!$PythonExecutable) { $PythonExecutable = (Get-Command python).Source }
& $PythonExecutable -c 'import numpy'
if ($LASTEXITCODE) { throw 'Full Windows contract tests require NumPy in -PythonExecutable.' }
Push-Location $repo
try {
  & $vcpkg install --triplet $Triplet "--overlay-triplets=$triplets" --x-feature=tests "--x-install-root=$DependencyRoot" "--x-buildtrees-root=$support/buildtrees" "--x-packages-root=$support/packages" "--downloads-root=$downloads"
  if ($LASTEXITCODE) { throw 'Windows dependency build failed' }
  $gpu = if ($Cuda) { 'ON' } else { 'OFF' }
  if ($Cuda -and !$CudaArchitectures) { throw 'CUDA builds require explicit -CudaArchitectures, e.g. 89.' }
  $options = @('-S', $repo, '-B', $BuildRoot, '-G', $Generator, '-A', 'x64',
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot/scripts/buildsystems/vcpkg.cmake",
    "-DVCPKG_TARGET_TRIPLET=$Triplet", "-DVCPKG_OVERLAY_TRIPLETS=$triplets",
    "-DVCPKG_INSTALLED_DIR=$DependencyRoot", '-DVCPKG_MANIFEST_INSTALL=OFF',
    "-DCMAKE_PREFIX_PATH=$QtRoot", '-DCOLITE_BUILD_GUI=ON', '-DCOLITE_BUILD_TESTS=ON',
    "-DPython3_EXECUTABLE=$PythonExecutable",
    '-DCOLITE_ENABLE_AVX2=OFF', '-DCOLITE_ENABLE_DELAUNAY_MESHING=OFF', '-DCOLITE_ENABLE_ONNX=OFF',
    "-DCOLITE_ENABLE_CUDA_SIFT=$gpu", "-DCOLITE_ENABLE_CUDA_MATCHER=$gpu", "-DCOLITE_ENABLE_CUDA_MVS=$gpu")
  if ($Cuda) {
    if (!$CudaToolkitRoot) { throw 'CUDA builds require -CudaToolkitRoot or CUDA_PATH.' }
    $options += @("-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures", '-T', "cuda=$CudaToolkitRoot")
  }
  & cmake @options
  if ($LASTEXITCODE) { throw 'Windows configure failed' }
  & cmake --build $BuildRoot --config Release --parallel $Jobs
  if ($LASTEXITCODE) { throw 'Windows build failed' }
  # Tests must resolve the same native dependencies as the installed worker.
  $previousPath = $env:PATH
  try {
    $env:PATH = "$QtRoot/bin;$DependencyRoot/$Triplet/bin;$previousPath"
    if ($Cuda) { $env:PATH = "$CudaToolkitRoot/bin/x64;$CudaToolkitRoot/bin;$env:PATH" }
    & ctest --test-dir $BuildRoot -C Release --output-on-failure
    if ($LASTEXITCODE) { throw 'Windows tests failed' }
  } finally { $env:PATH = $previousPath }
  & cmake --install $BuildRoot --config Release --prefix $InstallRoot
  if ($LASTEXITCODE) { throw 'Windows install failed' }
  $packageArgs = @{InstallRoot=$InstallRoot;QtRoot=$QtRoot;DependencyRoot=$DependencyRoot;Triplet=$Triplet}
  if ($Cuda) { $packageArgs.CudaRoot=$CudaToolkitRoot; $packageArgs.CudaArchitectures=$CudaArchitectures }
  & (Join-Path $PSScriptRoot 'package_windows.ps1') @packageArgs
  Write-Output $InstallRoot
} finally { Pop-Location }
