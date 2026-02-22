# run_orgs.ps1 — Launch all three org dashboards + FL server console
# Usage:  .\run_orgs.ps1

$env:PYTHONUTF8 = "1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  AURA -- Launching all org dashboards + FL console"   -ForegroundColor Cyan
Write-Host "======================================================"  -ForegroundColor Cyan
Write-Host ""

# Kill anything already on 8501-8504
foreach ($port in @(8501,8502,8503,8504)) {
    $results = netstat -ano | Select-String ":$port\s"
    foreach ($line in $results) {
        $p = ($line -split '\s+')[-1]
        if ($p -match '^\d+$') {
            Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Seconds 1

# Helper to launch a streamlit process in its own window
function Start-Dashboard($orgId, $script, $port, $color, $label) {
    $cmd = "`$env:PYTHONUTF8='1'; `$env:AURA_ORG_ID='$orgId'; & '$py' -m streamlit run $script --theme.base dark --server.address 0.0.0.0 --server.port $port --server.headless true"
    Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $cmd)
    Write-Host "  [$port] $label started" -ForegroundColor $color
    Start-Sleep -Milliseconds 900
}

Start-Dashboard "hospital"   "dashboard.py"           8501 "Green"  "Hospital   dashboard"
Start-Dashboard "bank"       "dashboard.py"           8502 "Yellow" "Bank        dashboard"
Start-Dashboard "university" "dashboard.py"           8503 "Blue"   "University  dashboard"
Start-Dashboard ""           "fl_server_dashboard.py" 8504 "Cyan"   "FL Server  console"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  All services running:"                                -ForegroundColor White
Write-Host "  Hospital   -> http://localhost:8501"                  -ForegroundColor Green
Write-Host "  Bank       -> http://localhost:8502"                  -ForegroundColor Yellow
Write-Host "  University -> http://localhost:8503"                  -ForegroundColor Blue
Write-Host "  FL Console -> http://localhost:8504"                  -ForegroundColor Cyan
Write-Host "======================================================"  -ForegroundColor Cyan
Write-Host ""
