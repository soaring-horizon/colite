param(
  [string]$QtRoot=$env:COLITE_QT_ROOT,
  [string]$BuildRoot='',
  [switch]$Test
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if(-not $QtRoot){$QtRoot=Join-Path $repo 'output/sdk/qt/5.15.2/msvc2019_64'}
if(-not $BuildRoot){$BuildRoot=Join-Path $repo 'output/gui_workbench/build'}
if(-not (Test-Path -LiteralPath (Join-Path $QtRoot 'lib/cmake/Qt5/Qt5Config.cmake'))){throw 'Set -QtRoot to an exact Qt 5.15.2 MSVC x64 SDK.'}
& cmake -S $PSScriptRoot -B $BuildRoot -G 'Visual Studio 16 2019' -A x64 "-DCMAKE_PREFIX_PATH=$QtRoot"
if($LASTEXITCODE){throw 'Configure failed'}
& cmake --build $BuildRoot --config Release --parallel 2
if($LASTEXITCODE){throw 'Build failed'}
$release=Join-Path $BuildRoot 'Release'
foreach($name in @('Qt5Core.dll','Qt5Gui.dll','Qt5Widgets.dll','Qt5Concurrent.dll','Qt5Test.dll','libEGL.dll','libGLESv2.dll')){
  Copy-Item -LiteralPath (Join-Path $QtRoot "bin/$name") -Destination $release
}
foreach($group in @('platforms','imageformats')){New-Item -ItemType Directory -Force (Join-Path $release $group) | Out-Null}
Copy-Item -LiteralPath (Join-Path $QtRoot 'plugins/platforms/qwindows.dll') -Destination (Join-Path $release 'platforms')
Copy-Item -LiteralPath (Join-Path $QtRoot 'plugins/platforms/qoffscreen.dll') -Destination (Join-Path $release 'platforms')
Copy-Item -LiteralPath (Join-Path $QtRoot 'plugins/imageformats/qjpeg.dll') -Destination (Join-Path $release 'imageformats')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'workflow/worker_bridge.py') -Destination $release
if($Test){& ctest --test-dir $BuildRoot -C Release --output-on-failure;if($LASTEXITCODE){throw 'GUI contract tests failed'}}
Write-Output (Join-Path $release 'colite_gui.exe')
