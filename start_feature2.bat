@echo off
title CrowdIQ - Feature 2: Gender-Aware Queues (port 8003)
cd /d "%~dp0feature2"
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --log-level info
pause
