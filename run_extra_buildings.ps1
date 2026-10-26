# Overnight run for the demo's extra "New building" candidates: train 3 buildings, then evaluate them.
#
# Run from the project root with the venv active (close the Streamlit demo first, and keep the laptop plugged in):
#     powershell -ExecutionPolicy Bypass -File .\run_extra_buildings.ps1
#
# Training comes first because the demo needs the models; the evaluation is the bonus. Both are logged to logs\.
# src/config.py is not touched. Nothing existing is overwritten: train_extra.py refuses if a model already exists.

$buildings = @("UnivClass_Nicholas", "Office_Emerald", "PrimClass_Uma")

New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = Join-Path logs ("extra_buildings_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".log")
function Write-Stamp($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line
}

# Keep Windows awake while this script runs (the screen may still turn off).
try {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Write-Stamp "Sleep prevention on"
} catch {
    Write-Stamp "Could not block sleep ($_). Set Sleep to 'Never' in Windows power settings instead."
}

Write-Stamp ("TRAIN: " + ($buildings -join ", "))
python scripts/train_extra.py @buildings 2>&1 | Tee-Object -FilePath $log -Append

Write-Stamp ("EVALUATE: " + ($buildings -join ", "))
python scripts/evaluation.py --buildings @buildings 2>&1 | Tee-Object -FilePath $log -Append

Write-Stamp "DONE. Log: $log"
try { [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }   # ES_CONTINUOUS only: allow sleep again
