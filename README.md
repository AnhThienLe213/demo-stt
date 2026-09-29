# Vietnamese Live Speech Recognition

Local FastAPI/WebSocket demo with PhoWhisper, Gipformer, and Zipformer model selection.

## Run locally

Use Python 3.10 or newer. The first startup downloads the model files and may take several minutes.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` and allow microphone access.

## Share through ngrok

Install ngrok and configure its authtoken once using the ngrok CLI. Keep the token out of this repository. In a second terminal, run:

```powershell
ngrok http 7860
```

Share the HTTPS forwarding URL printed by ngrok. Keep both the app and ngrok running while others use the demo.
