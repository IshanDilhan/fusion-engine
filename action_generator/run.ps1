# Action Generator PowerShell Runner
param (
    [string]$Step = "all"
)

$PythonExe = "C:\Users\IshanDilhan\AppData\Local\Programs\Python\Python311\python.exe"

Set-Location $PSScriptRoot

& $PythonExe run_all.py --step $Step
