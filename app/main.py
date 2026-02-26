from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import asyncio
import tempfile
import sys
import os

app = FastAPI()

@app.post("/run")
async def runChordExtraction(file: UploadFile = File(...)):
    print("API Endpoint reached.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".mp3":
        raise HTTPException(status_code=400, detail="Invalid file type {ext}")

    tempFile = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False
        ) as tmp:
            tempFile = tmp.name
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)
        print("Running Model...")
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        import runModel
        result = await asyncio.wait_for(
            run_in_threadpool(runModel.main, tempFile),
            timeout=120.0
        )
        print("Model Completed.")

        if result is None:
            raise HTTPException(status_code=500, detail="Chord Extraction Model failed to produce a result")

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chord Extraction Model failed: {str(e)}")

    finally:
        if tempFile and os.path.exists(tempFile):
            os.remove(tempFile)
