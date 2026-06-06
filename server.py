import base64
import os
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core import (
    SYSTEM_PROMPT,
    ConversationHistory,
    create_whisper_model,
    make_llm_client,
    chat_completion,
    summarize_user_utterances,
    transcribe,
    voicevox_synthesize,
)

app = FastAPI()

model = None
llm_client = None
history = None
lock = threading.Lock()


@app.on_event("startup")
def startup():
    global model, llm_client, history
    model = create_whisper_model()
    llm_client = make_llm_client()
    history = ConversationHistory(SYSTEM_PROMPT)


@app.post("/api/converse")
def converse(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio.file.read())
        tmp_path = f.name

    try:
        with lock:
            text = transcribe(model, tmp_path)
            if not text:
                return JSONResponse({"transcript": "", "reply": None, "audio_b64": None})

            reply = chat_completion(llm_client, history, text)
            wav_bytes = voicevox_synthesize(reply)
            audio_b64 = base64.b64encode(wav_bytes).decode()

        return JSONResponse({"transcript": text, "reply": reply, "audio_b64": audio_b64})
    finally:
        os.unlink(tmp_path)


@app.post("/api/summary")
def summary():
    with lock:
        text = summarize_user_utterances(llm_client, history)
    return JSONResponse({"summary": text})


@app.post("/api/reset")
def reset():
    with lock:
        history.clear()
    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
