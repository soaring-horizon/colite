param(
  [Parameter(Mandatory)][string]$InstallRoot,
  [Parameter(Mandatory)][string]$QtRoot,
  [Parameter(Mandatory)][string]$DependencyRoot,
  [string]$Triplet = 'x64-windows-colite',
  [string]$CudaRoot = '',
  [string]$CudaArchitectures = '',
  [string]$OnnxRoot = '',
  [string]$VcRuntimeRoot = ''
)
$ErrorActionPreference = 'Stop'
$release = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot '../version.json') | ConvertFrom-Json
if ($release.version -notmatch '^\d+\.\d+\.\d+$' -or $release.prerelease -notmatch '^[a-zA-Z0-9.]*$') { throw 'Invalid version.json' }
$releaseVersion = $release.version
if ($release.prerelease) { $releaseVersion += '-' + $release.prerelease }
$InstallRoot = (Resolve-Path -LiteralPath $InstallRoot).Path
$QtRoot = (Resolve-Path -LiteralPath $QtRoot).Path
$DependencyRoot = (Resolve-Path -LiteralPath $DependencyRoot).Path
$bin = Join-Path $InstallRoot 'bin'
if (!(Test-Path -LiteralPath (Join-Path $bin 'colite.exe')) -or
    !(Test-Path -LiteralPath (Join-Path $bin 'colite_gui.exe'))) {
  throw 'Install the native GUI and worker before packaging.'
}
$oldPath = $env:PATH
try {
  $env:PATH = "$QtRoot/bin;$oldPath"
  $version = & (Join-Path $QtRoot 'bin/qmake.exe') -query QT_VERSION
  if ($version.Trim() -ne '5.15.2') { throw 'Deployment requires exactly Qt 5.15.2' }
  & (Join-Path $QtRoot 'bin/windeployqt.exe') --release --no-plugins --no-translations --no-compiler-runtime --no-opengl-sw (Join-Path $bin 'colite_gui.exe')
  if ($LASTEXITCODE) { throw 'Qt runtime deployment failed' }
} finally { $env:PATH = $oldPath }
foreach ($plugin in @('platforms/qwindows.dll', 'imageformats/qjpeg.dll')) {
  $destination = Join-Path $bin $plugin
  New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
  Copy-Item -LiteralPath (Join-Path $QtRoot "plugins/$plugin") -Destination $destination -Force
}
if ((Get-Item -LiteralPath (Join-Path $bin 'Qt5Core.dll')).VersionInfo.FileVersion -notlike '5.15.2*') {
  throw 'Deployed Qt runtime does not match the SDK'
}
Get-ChildItem -LiteralPath (Join-Path $DependencyRoot "$Triplet/bin") -Filter '*.dll' |
  Where-Object { $_.Name -notmatch '^g(test|mock)' } |
  Copy-Item -Destination $bin -Force
if ($OnnxRoot) {
  $OnnxRoot = (Resolve-Path -LiteralPath $OnnxRoot).Path
  Copy-Item -LiteralPath (Join-Path $OnnxRoot 'lib/onnxruntime.dll') -Destination $bin -Force
  # This deployment path deliberately supports the CPU SDK. CUDA providers
  # additionally require an explicitly audited cuDNN/provider runtime closure.
  if (Test-Path -LiteralPath (Join-Path $OnnxRoot 'lib/onnxruntime_providers_cuda.dll')) {
    throw 'Use the CPU ONNX SDK for this packaging path; CUDA provider deployment needs its own runtime closure.'
  }
  if (!$VcRuntimeRoot) { throw 'ONNX packages require -VcRuntimeRoot for a compatible app-local MSVC runtime.' }
}
if ($VcRuntimeRoot) {
  $VcRuntimeRoot = (Resolve-Path -LiteralPath $VcRuntimeRoot).Path
  foreach ($required in @('msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    if (!(Test-Path -LiteralPath (Join-Path $VcRuntimeRoot $required))) { throw "Missing runtime $required" }
  }
  Get-ChildItem -LiteralPath $VcRuntimeRoot -Filter '*.dll' | Copy-Item -Destination $bin -Force
}
# Inspect imports with an explicit search path, never a machine's unrelated SDKs.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$dumpbin = & $vswhere -latest -products '*' -find 'VC/Tools/MSVC/**/bin/Hostx64/x64/dumpbin.exe' | Select-Object -First 1
if (!$dumpbin) { throw 'Visual Studio dumpbin is required to verify runtime closure' }
$search = @($bin, (Join-Path $QtRoot 'bin'), (Join-Path $DependencyRoot "$Triplet/bin"))
if ($CudaRoot) { $search += @((Join-Path $CudaRoot 'bin/x64'), (Join-Path $CudaRoot 'bin')) }
$queue = [Collections.Generic.Queue[string]]::new()
Get-ChildItem -LiteralPath $bin -Recurse -File | Where-Object { $_.Extension -in @('.exe', '.dll') } | ForEach-Object {$queue.Enqueue($_.FullName)}
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
while ($queue.Count) {
  $binary = $queue.Dequeue()
  if (!$seen.Add($binary)) { continue }
  $imports = & $dumpbin /nologo /dependents $binary
  if ($LASTEXITCODE) { throw "Cannot inspect runtime imports: $binary" }
  foreach ($line in $imports) {
    if ($line -notmatch '^\s+([^\s]+\.dll)\s*$') { continue }
    $name = $Matches[1]
    if ($name -match '^(api|ext)-ms-win-' -or (Test-Path -LiteralPath (Join-Path $env:WINDIR "System32/$name"))) { continue }
    $runtime = $null
    foreach ($folder in $search) {
      $candidate = Join-Path $folder $name
      if (Test-Path -LiteralPath $candidate) { $runtime = $candidate; break }
    }
    if (!$runtime) { throw "Unresolved runtime $name required by $binary" }
    $deployed = Join-Path $bin $name
    if ([IO.Path]::GetFullPath($runtime) -ne [IO.Path]::GetFullPath($deployed)) { Copy-Item -LiteralPath $runtime -Destination $deployed -Force }
    $queue.Enqueue($deployed)
  }
}
[IO.File]::WriteAllText((Join-Path $bin 'qt.conf'), "[Paths]`nPrefix=.`nPlugins=.`n", [Text.UTF8Encoding]::new($false))
$licenses = Join-Path $InstallRoot 'share/colite/licenses/runtime'
New-Item -ItemType Directory -Force $licenses | Out-Null
if ($OnnxRoot) {
  foreach ($name in @('LICENSE', 'ThirdPartyNotices.txt')) {
    Copy-Item -LiteralPath (Join-Path $OnnxRoot $name) -Destination (Join-Path $licenses ("ONNX-Runtime-" + $name)) -Force
  }
}
if ($CudaRoot) {
  $cudaLicenseCopied = $false
  foreach ($name in @('EULA.txt', 'LICENSE', 'CUDA_Toolkit_Release_Notes.txt', 'version.json')) {
    $notice = Join-Path $CudaRoot $name
    if (Test-Path -LiteralPath $notice) {
      Copy-Item -LiteralPath $notice -Destination (Join-Path $licenses ("NVIDIA-CUDA-" + $name)) -Force
      if ($name -in @('EULA.txt', 'LICENSE')) { $cudaLicenseCopied = $true }
    }
  }
  if (!$cudaLicenseCopied) { throw 'CUDA runtime license is missing from the supplied toolkit' }
}
Get-ChildItem -LiteralPath (Join-Path $DependencyRoot "$Triplet/share") -Directory | ForEach-Object {
  $copyright = Join-Path $_.FullName 'copyright'
  if (Test-Path -LiteralPath $copyright) {
    Copy-Item -LiteralPath $copyright -Destination (Join-Path $licenses ($_.Name + '-copyright.txt'))
  }
}
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../vcpkg.json') -Raw | ConvertFrom-Json
$runtime = @(Get-ChildItem -LiteralPath $bin -Recurse -File | ForEach-Object { [ordered]@{file=[IO.Path]::GetRelativePath($bin, $_.FullName); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
$identity = [ordered]@{schema=1; version=$releaseVersion; platform='windows-x64'; qt='5.15.2'; triplet=$Triplet; vcpkg_baseline=$manifest.'builtin-baseline'; cuda=[bool]$CudaRoot; cuda_architectures=$CudaArchitectures; source_frozen=$false; runtime=$runtime}
$identity | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $InstallRoot 'package-identity.json') -Encoding utf8
Write-Output $InstallRoot
