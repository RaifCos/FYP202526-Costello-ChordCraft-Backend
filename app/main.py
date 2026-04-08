from certifi import contents
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pathlib import Path
import tempfile
import asyncio
import magic
import json
import sys
import os

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Handle Model Importation.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import runModel

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

@app.post("/run")
@limiter.limit("5/minute")  # Rate Limit of 5 Requests per Minute.
async def runChordExtraction(request: Request, file: UploadFile = File(...)):
    print("API Endpoint reached.")
    ext = os.path.splitext(file.filename)[1].lower()

    # Sanitize filename.
    safeName = Path(file.filename).name
    if not safeName:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")
    
    # Only Accept MP3 and WAV Files.
    if ext != ".mp3" and ext != ".wav":
        raise HTTPException(status_code=400, detail=f"{ext} is an invalid file type, chord-CNN-LSTM only accepts .MP3 or .WAV files.")
    
    # Check MIME type to prevent malicious files.
    contents = await file.read(2048)
    await file.seek(0)
    mimeType = magic.from_buffer(contents, mime=True)

    # common audio MIME variants + fallback for some clients sending binary as octet-stream
    valid_mime_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "application/octet-stream"]
    if mimeType not in valid_mime_types:
        raise HTTPException(status_code=400, detail=f"{mimeType} is an invalid MIME type, chord-CNN-LSTM only accepts .MP3 or .WAV files.")
    
    print("Writing Temporary File...")
    tempFile = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False
        ) as tmp:
            # Make sure file doesn't exceed limit. 
            totalSize = 0
            tempFile = tmp.name
            try:
                # Timeout audio loading after 30 seconds. 
                async with asyncio.timeout(30):
                    while chunk := await file.read(1024 * 1024):
                        totalSize += len(chunk)
                        if totalSize > MAX_FILE_SIZE:   
                            raise HTTPException(status_code=400, detail="File size exceeds 50 MB limit.")
                        tmp.write(chunk)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="File upload timed out")
            
        # Run chord-CNN-LSTM Model.        
        print("Running Model...")
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        
        result = await asyncio.wait_for(
            run_in_threadpool(runModel.main, tempFile),
            timeout=30.0
        )

        print("Model Completed.")
        
        if result is None:
            raise HTTPException(status_code=500, detail="chord-CNN-LSTM Model failed to produce a result.")
        
        # Model Successfully produces a result! 
        if isinstance(result, str):
            result = json.loads(result)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Model timed out after 120 seconds.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chord-CNN-LSTM Model Error: {str(e)}")
    
    finally:
        if tempFile and os.path.exists(tempFile):
            os.remove(tempFile)
