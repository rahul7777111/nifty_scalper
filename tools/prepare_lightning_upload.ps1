param(
    [string]$OutputDir = "dist/lightning_upload_bundle"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$destinationRoot = Join-Path $repoRoot $OutputDir

$includePaths = @(
    "src",
    "scripts",
    "tests",
    "tools",
    "docs",
    "config",
    "pytradingapi-typeB-main",
    "DhanHQ-py-main",
    "README.md",
    "SETTINGS_GUIDE.md",
    "requirements.txt",
    "pyproject.toml",
    "pytest.ini",
    "conftest.py",
    ".env.example",
    "Procfile",
    "railway.toml",
    "AGENTS.md"
)

$excludeDirNames = @(
    ".git",
    ".venv",
    ".venv_linux",
    ".venv_broken_uv_tcl",
    ".uv-cache",
    ".uv-python",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    "data",
    "models",
    "reports",
    "logs",
    "artifacts",
    "archive",
    "backups",
    "terminals",
    "myenv",
    "Documents",
    "C_tmp_test_artifact"
)

$excludeFileNames = @(
    ".env",
    ".scalper.env",
    ".engine_diagnostics.json",
    ".historical_expansion.log",
    ".last_timestamp.txt",
    ".scalper.ui.log",
    "desktop.ini",
    "trades.db",
    "ui_stdout.log",
    "ui_stderr.log",
    "retrain.log",
    "retrain.prof",
    "retrain_run.log",
    "mconnect.log",
    "ml_signal_model.pkl",
    "ml_signal_model.pkl.QUARANTINED_45samples_overfit"
)

$excludeExtensions = @(
    ".db",
    ".log",
    ".prof"
)

function Remove-IfExists {
    param([string]$PathValue)

    if (Test-Path -LiteralPath $PathValue) {
        Remove-Item -LiteralPath $PathValue -Recurse -Force
    }
}

function Should-SkipFile {
    param([System.IO.FileInfo]$FileInfo)

    if ($excludeFileNames -contains $FileInfo.Name) {
        return $true
    }
    if ($excludeExtensions -contains $FileInfo.Extension.ToLowerInvariant()) {
        return $true
    }
    if ($FileInfo.Name -match "^_.*\.(py|json|txt)$") {
        return $true
    }
    if ($FileInfo.Name -match "^tmp_.*") {
        return $true
    }
    return $false
}

function Copy-IncludedPath {
    param(
        [string]$SourceRelativePath
    )

    $sourcePath = Join-Path $repoRoot $SourceRelativePath
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        return
    }

    $item = Get-Item -LiteralPath $sourcePath
    $destinationPath = Join-Path $destinationRoot $SourceRelativePath

    if ($item.PSIsContainer) {
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
        $entries = Get-ChildItem -LiteralPath $sourcePath -Force
        foreach ($entry in $entries) {
            if ($entry.PSIsContainer) {
                if ($excludeDirNames -contains $entry.Name) {
                    continue
                }
                Copy-IncludedPath -SourceRelativePath (Join-Path $SourceRelativePath $entry.Name)
                continue
            }
            if (Should-SkipFile -FileInfo $entry) {
                continue
            }
            $destFile = Join-Path $destinationPath $entry.Name
            Copy-Item -LiteralPath $entry.FullName -Destination $destFile -Force
        }
        return
    }

    $parentDir = Split-Path -Parent $destinationPath
    if ($parentDir) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    if (-not (Should-SkipFile -FileInfo $item)) {
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
}

Remove-IfExists -PathValue $destinationRoot
New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null

foreach ($path in $includePaths) {
    Copy-IncludedPath -SourceRelativePath $path
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    source_repo = $repoRoot
    output_dir = $destinationRoot
    included_paths = $includePaths
    excluded_directory_names = $excludeDirNames
    excluded_file_names = $excludeFileNames
    excluded_extensions = $excludeExtensions
}

$manifestPath = Join-Path $destinationRoot "LIGHTNING_UPLOAD_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Prepared Lightning upload bundle at: $destinationRoot"
Write-Host "Next step:"
Write-Host "  lightning upload `"$destinationRoot`" --studio <teamspace/studio_name> --recursive"
