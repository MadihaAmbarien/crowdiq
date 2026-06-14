# CrowdIQ - Master Startup Script
$root = "C:\Users\hp\Desktop\major ke docs\Asal nagad project\CrowdIQ (2) Wasil\CrowdIQ"

Write-Host ""
Write-Host "  =====================================" -ForegroundColor Cyan
Write-Host "   CrowdIQ - Starting All Services" -ForegroundColor Cyan
Write-Host "  =====================================" -ForegroundColor Cyan
Write-Host ""

$features = @(
    @{name="F1-QueueMgmt";     dir="feature1"; port=8002},
    @{name="F2-GenderQueue";   dir="feature2"; port=8003},
    @{name="F3-PriorityQueue"; dir="feature3"; port=8004},
    @{name="F5-Voice";         dir="feature5"; port=8005},
    @{name="F6-Emergency";     dir="feature6"; port=8006},
    @{name="F7-Heatmap";       dir="feature7"; port=8007},
    @{name="F8-Emotion";       dir="feature8"; port=8008},
    @{name="F9-Predictive";    dir="feature9"; port=8009}
)

foreach ($f in $features) {
    $dir = Join-Path $root $f.dir
    Write-Host "  Starting $($f.name) on port $($f.port)..." -ForegroundColor Yellow
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/k title $($f.name) && python -m uvicorn main:app --host 0.0.0.0 --port $($f.port) --log-level warning" `
        -WorkingDirectory $dir `
        -WindowStyle Minimized
    Start-Sleep -Milliseconds 300
}

Write-Host ""
Write-Host "  All backends launched. Waiting 8s for initialization..." -ForegroundColor Gray
Start-Sleep -Seconds 8

Write-Host "  Starting Frontend (Next.js on port 3000)..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/k title CrowdIQ-Frontend && npm run dev" `
    -WorkingDirectory "$root\frontend" `
    -WindowStyle Normal

Write-Host "  Waiting 15s for Next.js to compile..." -ForegroundColor Gray
Start-Sleep -Seconds 15

Write-Host "  Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "  =====================================" -ForegroundColor Green
Write-Host "   CrowdIQ is LIVE at localhost:3000" -ForegroundColor Green
Write-Host "  =====================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Backends running (minimized in taskbar):" -ForegroundColor Gray
Write-Host "    Feature 1 (Queue+Video)  -> :8002" -ForegroundColor Gray
Write-Host "    Feature 2 (Gender Queue) -> :8003" -ForegroundColor Gray
Write-Host "    Feature 3 (Priority)     -> :8004" -ForegroundColor Gray
Write-Host "    Feature 5 (Voice)        -> :8005" -ForegroundColor Gray
Write-Host "    Feature 6 (Emergency)    -> :8006" -ForegroundColor Gray
Write-Host "    Feature 7 (Heatmap)      -> :8007" -ForegroundColor Gray
Write-Host "    Feature 8 (Emotion)      -> :8008" -ForegroundColor Gray
Write-Host "    Feature 9 (Predictive)   -> :8009" -ForegroundColor Gray
Write-Host ""
