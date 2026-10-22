[CmdletBinding()]
param(
    [string]$Python = '.venv-cpu\Scripts\python.exe',
    [string]$ModelsZip = '', [string]$ModelsSha256 = '',
    [string]$EvidenceZip = '', [string]$EvidenceSha256 = '',
    [string]$OutputDirectory = '',
    [switch]$Offline,
    [switch]$SkipPublicResources
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $projectRoot $Python }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if ([bool]$ModelsZip -ne [bool]$ModelsSha256) { throw 'Pass ModelsZip and ModelsSha256 together.' }
if ([bool]$EvidenceZip -ne [bool]$EvidenceSha256) { throw 'Pass EvidenceZip and EvidenceSha256 together.' }
if (-not $OutputDirectory) { $OutputDirectory = 'runs\restore_' + [guid]::NewGuid().ToString('N') }
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot $OutputDirectory }
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a new OutputDirectory to preserve previous restoration records.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
function Invoke-Checked([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Resource preparation failed with exit code $LASTEXITCODE" }
}
if (-not $SkipPublicResources) {
    $resourceArguments = @((Join-Path $PSScriptRoot 'prepare_resources.py'), '--output-dir', (Join-Path $OutputDirectory 'public_resources'))
    if ($Offline) { $resourceArguments += '--offline' }
    Invoke-Checked $resourceArguments
}
foreach ($artifact in @(@{path=$ModelsZip; hash=$ModelsSha256; name='models'}, @{path=$EvidenceZip; hash=$EvidenceSha256; name='evidence'})) {
    if ($artifact.path) {
        Invoke-Checked @((Join-Path $PSScriptRoot 'restore_artifact.py'), '--zip', $artifact.path, '--sha256', $artifact.hash,
            '--output', (Join-Path $OutputDirectory ($artifact.name + '.json')))
    }
}
Write-Output "Restoration completed; report directory: $OutputDirectory"
