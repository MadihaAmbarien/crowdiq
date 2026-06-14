@echo off
title CrowdIQ - Feature 6: Emergency Alert & Response (port 8006)
cd /d "%~dp0feature6"
python -m uvicorn main:app --host 0.0.0.0 --port 8006 --log-level info
pause
