from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess

app = FastAPI()

# Allow JS to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

cli_process = None
RAAGAS = ["bhairav", "bhairavi", "bhupali", "malkauns", "asavari"]

# Serve HTML
@app.get("/")
def index():
    return FileResponse("static/index.html")

# Raaga list
@app.get("/get_raagas")
def get_raagas():
    return JSONResponse({"raagas": RAAGAS})

# Start CLI
@app.get("/start_cli")
def start_cli(raag: str):
    global cli_process
    if cli_process:
        return JSONResponse({"message": "Already running"})
    if raag not in RAAGAS:
        return JSONResponse({"message": f"Raaga '{raag}' not found"}, status_code=400)

    try:
        python_cmd = "python3"
        cli_process = subprocess.Popen([python_cmd, "Techsen_CLI.py", "-r", raag])
        return JSONResponse({"message": f"Started CLI with Raaga {raag}"})
    except Exception as e:
        return JSONResponse({"message": str(e)}, status_code=500)

# Stop CLI
@app.get("/stop_cli")
def stop_cli():
    global cli_process
    if cli_process:
        cli_process.terminate()
        cli_process = None
        return JSONResponse({"message": "CLI stopped"})
    return JSONResponse({"message": "CLI was not running"})
