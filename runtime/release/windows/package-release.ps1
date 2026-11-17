param([switch]$GenerateUninstaller)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$out = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../target/release-windows'))
$stage = Join-Path $out 'stage'
$version = (Get-Content "$out/version.txt" -Raw).Trim()
$makensis = Join-Path ${env:ProgramFiles(x86)} 'NSIS/makensis.exe'
if (!(Test-Path $makensis)) { throw "NSIS not found: $makensis" }

if ($GenerateUninstaller) {
    $helper = Join-Path $out 'generate-uninstaller.exe'
    & $makensis /V2 "/DVERSION=$version" "/DSTAGE=$stage" "/DOUTFILE=$helper" /DGENERATE_UNINSTALLER "$PSScriptRoot/installer.nsi"
    if ($LASTEXITCODE -ne 0) { throw 'Uninstaller generator compilation failed' }
    $process = Start-Process -FilePath $helper -ArgumentList '/S' -Wait -PassThru
    if ($process.ExitCode -ne 0 -or !(Test-Path "$stage/Uninstall.exe")) {
        throw 'Uninstaller generation failed'
    }
} else {
    # Do not produce a release installer unless all embedded binaries are signed.
    & "$PSScriptRoot/verify-signatures.ps1" -Path $stage
    $setup = Join-Path $out "2ndPressing-$version-windows-x64-setup.exe"
    & $makensis /V2 "/DVERSION=$version" "/DSTAGE=$stage" "/DOUTFILE=$setup" /DSIGNED_UNINSTALLER "$PSScriptRoot/installer.nsi"
    if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed' }
}
