[CmdletBinding()]
param(
    [ValidateSet('Cpu', 'Cuda126', 'DataOnly')][string]$Mode = 'Cpu',
    [string]$Python = '',
    [string]$EnvironmentPath = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $EnvironmentPath) {
    $EnvironmentPath = if ($Mode -eq 'DataOnly') { '.venv' } else { '.venv-' + $Mode.ToLowerInvariant() }
}
if (-not [IO.Path]::IsPathRooted($EnvironmentPath)) { $EnvironmentPath = Join-Path $projectRoot $EnvironmentPath }
$envDirectory = [IO.Path]::GetFullPath($EnvironmentPath)
$rootPrefix = [IO.Path]::GetFullPath($projectRoot).TrimEnd('\') + '\'
if (-not $envDirectory.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'EnvironmentPath must be inside this checkout; choose a new project-local environment.'
}
$envPython = Join-Path $envDirectory 'Scripts\python.exe'
$marker = Join-Path $envDirectory '.flow_contact_environment.json'
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $Program" }
}
function Test-Runtime {
    Invoke-Checked $envPython @('-c', 'import sys; assert sys.version_info[:2] == (3,12), sys.version; print(sys.version)')
    Invoke-Checked $envPython @('-m', 'pip', 'check')
    $runtimeCheck = @'
import collections, importlib.metadata as md, json, sysconfig, sys
from pathlib import Path
import numpy, pymunk, gymnasium, av, pyarrow, pygame, shapely
groups = collections.defaultdict(list)
for item in md.distributions(path=[sysconfig.get_paths()['purelib']]):
    groups[item.metadata['Name'].lower().replace('_', '-')].append(item.version)
duplicates = {key: value for key, value in groups.items() if len(value) > 1}
assert not duplicates, duplicates
for line in Path(sys.argv[1]).read_text().splitlines():
    if '==' not in line or line.startswith('#'):
        continue
    name, expected = line.split('==', 1)
    if name == 'torch':
        continue  # CPU/CUDA wheel flavor is selected independently.
    assert md.version(name) == expected, (name, md.version(name), expected)
print(json.dumps({'duplicate_package_metadata': duplicates, 'locked_versions': 'passed', 'data_runtime': 'passed'}))
'@
    $checkLock = if ($Mode -eq 'DataOnly') { 'requirements-data.lock.txt' } else { 'requirements-training.lock.txt' }
    Invoke-Checked $envPython @('-c', $runtimeCheck, (Join-Path $projectRoot $checkLock))
    if ($Mode -ne 'DataOnly') {
        $torchCheck = @'
import torch, diffusers, einops, json
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert diffusers.__version__ == '0.35.1'
assert einops.__version__ == '0.8.1'
x = torch.tensor([1., 2.], requires_grad=True); (x.square().sum()).backward()
print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cpu_forward_backward': 'passed'}))
'@
        Invoke-Checked $envPython @('-c', $torchCheck)
        if ($Mode -eq 'Cuda126') {
            $cudaCheck = "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable; use Cpu mode'; x=torch.ones(2, device='cuda', requires_grad=True); x.square().sum().backward(); print(torch.cuda.get_device_name(0))"
            Invoke-Checked $envPython @('-c', $cudaCheck)
        }
    }
}
if ($CheckOnly) {
    if (-not (Test-Path -LiteralPath $envPython -PathType Leaf)) { throw "Environment Python missing: $envPython" }
    Test-Runtime
    Write-Output 'Existing runtime checked; no environment was created or installed.'
    exit 0
}
if (Test-Path -LiteralPath $envDirectory) {
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "Existing environment has no setup marker; preserved. Choose a new -EnvironmentPath or use -CheckOnly: $envDirectory"
    }
    $oldMarker = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    if ($oldMarker.mode -ne $Mode) { throw 'Refusing to change the mode of an existing environment.' }
}
$temporaryDirectory = Join-Path $projectRoot ('tmp\setup_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
try {
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    if (-not (Test-Path -LiteralPath $envPython -PathType Leaf)) {
        $baseArguments = @()
        if ($Python) { $basePython = $Python }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { $basePython = 'py'; $baseArguments = @('-3.12') }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { $basePython = 'python' }
        else { throw 'Install Python 3.12 first, or pass -Python with its executable path.' }
        Invoke-Checked $basePython ($baseArguments + @('-c', 'import sys; assert sys.version_info[:2] == (3,12), sys.version; print(sys.executable)'))
        Invoke-Checked $basePython ($baseArguments + @('-m', 'venv', $envDirectory))
    }
    @{ mode = $Mode; status = 'installing'; python = '3.12'; created_by = 'scripts/setup_windows.ps1' } |
        ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    Invoke-Checked $envPython @('-c', 'import sys; assert sys.version_info[:2] == (3,12), sys.version')
    if ($Mode -eq 'DataOnly') {
        Invoke-Checked $envPython @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements-data.lock.txt'))
    } else {
        $wheelIndex = if ($Mode -eq 'Cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu126' }
        # Install one Torch wheel without dependency resolution, then the exact remaining lock in one transaction.
        # This avoids first installing unpinned Torch dependencies and replacing their metadata immediately afterwards.
        Invoke-Checked $envPython @('-m', 'pip', 'install', '--no-deps', 'torch==2.7.1', '--index-url', $wheelIndex)
        $portableLock = Join-Path $temporaryDirectory 'requirements-without-torch.lock.txt'
        Get-Content -LiteralPath (Join-Path $projectRoot 'requirements-training.lock.txt') |
            Where-Object { $_ -notmatch '^torch==' } | Set-Content -LiteralPath $portableLock -Encoding ASCII
        Invoke-Checked $envPython @('-m', 'pip', 'install', '-r', $portableLock)
    }
    Test-Runtime
    @{ mode = $Mode; status = 'passed'; python = '3.12'; created_by = 'scripts/setup_windows.ps1' } |
        ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    Write-Output "Ready: $envPython"
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
