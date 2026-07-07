# build.ps1 — convenience wrapper that puts WinLibs GCC 16 on PATH and runs make.
# Usage:  .\build.ps1            (release build)
#         .\build.ps1 perft      (build + run perft self-test)
#         .\build.ps1 clean
param([Parameter(ValueFromRemainingArguments=$true)] [string[]] $Args)

$gccBin = "C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if (-not (Test-Path "$gccBin\g++.exe")) { Write-Error "WinLibs g++ not found at $gccBin"; exit 1 }
$env:Path = "$gccBin;$env:Path"

& "$gccBin\mingw32-make.exe" @Args
exit $LASTEXITCODE
