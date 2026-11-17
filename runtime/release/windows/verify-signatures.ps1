param([Parameter(Mandatory)][string]$Path)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$signtool = Get-ChildItem "${env:ProgramFiles(x86)}/Windows Kits/10/bin/*/x64/signtool.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1
if (!$signtool) { throw 'Windows SDK SignTool not found' }
$files = @(Get-ChildItem -LiteralPath $Path -File -Recurse |
    Where-Object Extension -in '.exe', '.dll', '.vst3', '.clap')
if ($files.Count -eq 0) { throw "No binaries found: $Path" }
foreach ($file in $files) {
    # /tw warns for missing timestamps; treat any warning as a failed release.
    & $signtool.FullName verify /pa /all /tw /v $file.FullName
    if ($LASTEXITCODE -ne 0) { throw "Signature/timestamp verification failed: $($file.FullName)" }
}
