@echo off
echo ===================================================
echo 🚀 Starting Store Intelligence (API + Dashboard) 🚀
echo ===================================================

echo Starting FastAPI Backend on Port 8000...
set DEMO_DATE=2026-04-10
set DB_PATH=data/store_intelligence.db
set POS_CSV_PATH=data/pos_transactions.csv
set STORE_LAYOUT_PATH=data/store_layout.json
start "Store Intelligence API" cmd /c "venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo Starting Frontend Dashboard on Port 3000...
start "Store Intelligence Dashboard" cmd /c "python -m http.server 3000 --directory dashboard"

echo.
echo ✅ Servers are running! 
echo 👉 Open http://localhost:3000 in your browser.
echo.
pause
