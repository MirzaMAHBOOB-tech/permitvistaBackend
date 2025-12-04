@echo off 
cd /d "D:\Tasks\permit-certs-project\" 
call "D:\Tasks\permit-certs-project\.venv\Scripts\activate.bat" 
echo [INFO] Running: python -m uvicorn api_server:app --reload --port 8000 
python -u -m uvicorn api_server:app --reload --port 8000 
