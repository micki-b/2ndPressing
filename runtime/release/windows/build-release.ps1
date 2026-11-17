# Native Windows build and staging for the signed release workflow.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location (Join-Path $PSScriptRoot '../..')

$metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$version = ($metadata.packages | Where-Object name -eq 'pressing-app').version
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw "Unsupported release version: $version" }
$out = Join-Path $PWD 'target/release-windows'
if (Test-Path $out) { throw "Output directory already exists: $out. Use a clean checkout." }
$stage = Join-Path $out 'stage'
$plugin = '2ndPressing Declicker'
$vst = Join-Path $stage "vst3/$plugin.vst3/Contents/x86_64-win"
New-Item -ItemType Directory -Force $vst, "$stage/app", "$stage/clap" | Out-Null

cargo build --locked --release --target x86_64-pc-windows-msvc -p pressing-plugin --lib
if ($LASTEXITCODE -ne 0) { throw 'Plugin build failed' }
cargo build --locked --release --target x86_64-pc-windows-msvc -p pressing-app
if ($LASTEXITCODE -ne 0) { throw 'App build failed' }
$bin = 'target/x86_64-pc-windows-msvc/release'
Copy-Item "$bin/pressing_plugin.dll" "$vst/$plugin.vst3"
Copy-Item "$bin/pressing_plugin.dll" "$stage/clap/$plugin.clap"
Copy-Item "$bin/2ndpressing.exe" "$stage/app/2ndPressing.exe"
Copy-Item "$PSScriptRoot/2ndPressing.ico" $stage
Copy-Item '../LICENSE' "$stage/LICENSE.txt"
Get-ChildItem -LiteralPath $stage -File -Recurse | Select-Object FullName, Length
Set-Content -LiteralPath "$out/version.txt" -Value $version -Encoding ascii
