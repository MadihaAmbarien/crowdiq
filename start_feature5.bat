@echo off
title CrowdIQ - Feature 5: AI-Powered Voice Assistance (port 8005)
cd /d "%~dp0feature5"
python -m uvicorn main:app --host 0.0.0.0 --port 8005 --log-level info
pause
