@echo off
title CrowdIQ - Feature 9: Predictive Analytics (port 8009)
cd /d "%~dp0feature9"
python -m uvicorn main:app --host 0.0.0.0 --port 8009 --log-level info
pause
