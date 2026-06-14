@echo off
title CrowdIQ - Feature 8: Emotion & Sentiment Analysis (port 8008)
cd /d "%~dp0feature8"
python -m uvicorn main:app --host 0.0.0.0 --port 8008 --log-level info
pause
