from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import tempfile
import json
import os

app = FastAPI()

ACR_SCRIPT = "scripts/modelLibrosa.py"

@app.post("/run")
async def runChordExtraction(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if not ext == ".mp3":
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False
        ) as tmp:
            tempFile = tmp.name
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

        result = subprocess.run(
            ["python", ACR_SCRIPT, tempFile],
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Chord Extraction Model timed out")

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Chord Extraction Model failed",
                "stderr": e.stderr
            }
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Chord Extraction Model did not return valid JSON"
        )

    finally:
        if "tempFile" in locals() and os.path.exists(tempFile):
            os.remove(tempFile)
