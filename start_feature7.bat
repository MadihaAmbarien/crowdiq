@echo off
title CrowdIQ - Feature 7: Crowd Density & Heatmap Analytics (port 8007)
cd /d "%~dp0feature7"
python -m uvicorn main:app --host 0.0.0.0 --port 8007 --log-level info
pause
