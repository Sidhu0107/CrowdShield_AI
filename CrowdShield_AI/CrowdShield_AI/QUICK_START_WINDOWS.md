# Quick Start: Windows (Backend + Frontend)

## 1) Backend (does NOT open webcam on startup)

```powershell
cd CrowdShield_AI
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$((Get-Location).Path)"
pip install -r backend/api/requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

This starts the API only. Camera/webcam stays idle until you click **Connect stream** in **Live Monitor**.

## 2) Frontend

```powershell
cd CrowdShield_AI\frontend
npm run dev
```

Open: `http://localhost:5173`

## 3) Live Monitor webcam control

- Go to **Live Monitor**.
- Select source (`Webcam 0` / `Webcam 1`).
- Click **Connect stream** to start webcam.
- Click **Disconnect stream** to stop webcam.

## 4) Analyze Video (offline upload)

Use **Analyze Video** page for file upload + analysis workflow.

## Optional: Pipeline tester (manual)

`test_pipeline.py` is only for local testing/debug visualization.

```powershell
python test_pipeline.py --source 0
```

If you run it without `--source`, it will abort and ask for explicit source.
