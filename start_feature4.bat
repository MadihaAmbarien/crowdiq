@echo off
title CrowdIQ - Feature 4: Smart QR Assistance (port 8004)
cd /d "%~dp0feature4"
python -m uvicorn main:app --host 0.0.0.0 --port 8004 --log-level info
pause
