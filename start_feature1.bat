@echo off
title CrowdIQ - Feature 1: Queue Management (port 8002)
cd /d "%~dp0feature1"
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --log-level info
pause
