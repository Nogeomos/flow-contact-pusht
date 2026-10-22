# Run only after the training process has finished. Preserve all replaced files.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot '.venv-train\Lib\site-packages')).Path
$runRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'runs\stages234_20261022')).Path
$backupRoot = Join-Path $runRoot 'runtime_repair_backup'
if (Test-Path -LiteralPath $backupRoot) { throw 'Repair backup already exists; inspect before any retry.' }
$packageNames = @('filelock','filelock-3.32.3.dist-info','filelock-4.2.4.dist-info',
                  'fsspec','fsspec-2026.7.0.dist-info','fsspec-2026.9.0.dist-info')
$resolvedTargets = @()
foreach ($packageName in $packageNames) {
    $targetPath = (Resolve-Path -LiteralPath (Join-Path $packageRoot $packageName)).Path
    if (-not $targetPath.StartsWith($packageRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escaped project environment: $targetPath"
    }
    if ((Split-Path -Parent $targetPath) -ne $packageRoot) { throw 'Unexpected nested target' }
    $resolvedTargets += $targetPath
}
$resolvedBackup = [IO.Path]::GetFullPath($backupRoot)
if (-not $resolvedBackup.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Backup escaped project workspace'
}
New-Item -ItemType Directory -Path $resolvedBackup | Out-Null
$resolvedTargets | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'runtime_repair_targets.json') -Encoding utf8
foreach ($targetPath in $resolvedTargets) {
    Move-Item -LiteralPath $targetPath -Destination $resolvedBackup
}
$trainingPython = Join-Path $projectRoot '.venv-train\Scripts\python.exe'
$wheelRoot = Join-Path $runRoot 'runtime_repair_wheels'
& $trainingPython -m pip install --no-index --no-deps --find-links $wheelRoot filelock==3.32.3 fsspec==2026.7.0
if ($LASTEXITCODE -ne 0) { throw 'Install failed; preserved original package folders in runtime_repair_backup.' }
& $trainingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $trainingPython (Join-Path $projectRoot 'research\audit_runtime_packages.py') --output (Join-Path $runRoot 'runtime_metadata_after_repair.json')
if ($LASTEXITCODE -ne 0) { throw 'Runtime audit failed.' }
$check = Get-Content -LiteralPath (Join-Path $runRoot 'runtime_metadata_after_repair.json') -Raw | ConvertFrom-Json
if ($check.duplicate_metadata.PSObject.Properties.Count -gt 0) { throw 'Duplicate distributions remain.' }
if (-not $check.all_selected_source_records_match) { throw 'Installed code differs from selected wheel records.' }
Write-Output 'REPAIR_VERIFIED: original active package versions restored; duplicate metadata removed; backup preserved.'
