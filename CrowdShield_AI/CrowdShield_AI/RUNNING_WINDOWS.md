# Running CrowdShield AI on Windows

This guide uses the API backend and React frontend. The backend does not auto-open webcam.

## Prerequisites

- Python 3.10+
- Node.js 20+
- Virtual environment recommended

## One-time setup

### Backend deps

```powershell
cd CrowdShield_AI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/api/requirements.txt
pip install -r requirements.txt
```

### Frontend deps

```powershell
cd CrowdShield_AI\frontend
npm install
```

## Start the app

### Terminal 1: API backend

```powershell
cd CrowdShield_AI
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$((Get-Location).Path)"
pip install -r backend/api/requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Frontend

```powershell
cd CrowdShield_AI\frontend
npm run dev
```

Frontend URL: `http://localhost:5173`

## Webcam behavior (important)

- Backend startup does **not** open webcam.
- Webcam starts only when Live Monitor sends `POST /api/live/start`.
- Webcam stops when Live Monitor sends `POST /api/live/stop`.

## Live Monitor usage

1. Open **Live Monitor**.
2. Choose source (`Webcam 0` or `Webcam 1`).
3. Click **Connect stream**.
4. Click **Disconnect stream** to release camera.

## Analyze Video usage

Use **Analyze Video** for upload + offline analysis.

## Optional tester script

`test_pipeline.py` is a debug tester and can open webcam if source is webcam.

```powershell
python test_pipeline.py --source 0
python test_pipeline.py --source stampede.mp4
```

## Quick troubleshooting

- If frontend cannot reach API: check backend is running on port `8000`.
- If camera fails to open: close other apps using webcam and try source `1`.
- If Python modules missing: activate `venv` and reinstall both `requirements.txt` and `backend/api/requirements.txt`.
