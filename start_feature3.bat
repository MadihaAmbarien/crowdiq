@echo off
title CrowdIQ - Feature 3: Priority Elderly & Disabled (port 8004)
cd /d "%~dp0feature3"
python -m uvicorn main:app --host 0.0.0.0 --port 8004 --log-level info
pause
