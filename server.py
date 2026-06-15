import asyncio
import base64
import json
import queue
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core import (
    SYSTEM_PROMPT,
    STT_HALLUCINATIONS,
    create_recorder,
    make_llm_client,
    chat_stream,
    summarize_utterances,
    voicevox_synthesize,
    split_complete_sentences,
)

app = FastAPI()

recorder = None
llm_client = None

# 現在アクティブな WebSocket セッション（単一ユーザー前提）
active: dict = {"loop": None, "queue": None}


def _on_realtime(text: str):
    """RealtimeSTT スレッドから asyncio キューへ暫定テキストを橋渡し。"""
    loop = active["loop"]
    q = active["queue"]
    if loop and q:
        loop.call_soon_threadsafe(q.put_nowait, {"type": "transcript_interim", "text": text})


@app.on_event("startup")
def startup():
    global recorder, llm_client
    llm_client = make_llm_client()
    print("RealtimeSTT レコーダーを初期化中 (use_microphone=False)...")
    recorder = create_recorder(_on_realtime, use_microphone=False)
    print("起動完了。ブラウザで http://127.0.0.1:8000 を開いてください。")


def _b64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode()


@app.websocket("/ws/converse")
async def ws_converse(ws: WebSocket):
    # 多重接続を防ぐ（単一ユーザー前提）
    if active["loop"] is not None:
        await ws.close(code=1008)
        return

    await ws.accept()
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue()
    # クライアントから generate メッセージを受け取るキュー（worker スレッドで .get() する）
    gen_q: queue.Queue = queue.Queue()
    state = {"connected": True, "paused": False, "sample_rate": 16000, "mode": "chat"}

    active["loop"] = loop
    active["queue"] = out_q

    def push(evt: dict):
        """スレッドセーフに WebSocket 送信キューへイベントを追加する。"""
        loop.call_soon_threadsafe(out_q.put_nowait, evt)

    # ── Worker スレッド（ブロッキング: STT確定 → LLM → TTS） ──────────────
    def worker():
        while state["connected"]:
            try:
                text = recorder.text()      # 話し終わるまでブロック; 間に _on_realtime が発火
            except Exception as e:
                if state["connected"]:
                    push({"type": "error", "message": str(e)})
                break

            if not state["connected"]:
                break
            if not text or not text.strip():
                continue
            if text.strip() in STT_HALLUCINATIONS:
                continue

            push({"type": "transcript_final", "text": text})

            if text.strip(" .") in ("リセット", "最初から", "リセットして"):
                push({"type": "reset"})
                continue

            # 書き起こしモード: LLM/TTS を呼ばずそのまま次の発話へ
            if state["mode"] == "memo":
                continue

            # 会話モード: クライアントが送ってくる表示中コンテキストを待つ
            try:
                messages = gen_q.get()      # {"type":"generate","messages":[...]} のメッセージ列
            except Exception:
                break
            if messages is None or not state["connected"]:
                break

            # system プロンプトを先頭に付けて LLM 呼出し
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

            try:
                buf = ""
                for delta in chat_stream(llm_client, full_messages):
                    push({"type": "reply_delta", "text": delta})
                    buf += delta
                    sents, buf = split_complete_sentences(buf)
                    for s in sents:
                        try:
                            push({"type": "audio", "b64": _b64(voicevox_synthesize(s))})
                        except Exception as e:
                            push({"type": "error", "message": str(e)})
                buf = buf.strip()
                if buf:
                    try:
                        push({"type": "audio", "b64": _b64(voicevox_synthesize(buf))})
                    except Exception as e:
                        push({"type": "error", "message": str(e)})
                push({"type": "done"})
            except Exception as e:
                push({"type": "error", "message": str(e)})
                push({"type": "done"})

    # ── Sender タスク（out_q → WebSocket 送信） ───────────────────────────
    async def sender():
        while True:
            evt = await out_q.get()
            if evt is None:
                break
            try:
                await ws.send_json(evt)
            except Exception:
                break

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    sender_task = asyncio.create_task(sender())

    # ── 受信ループ（ブラウザ PCM + 制御メッセージ） ────────────────────────
    try:
        while True:
            msg = await ws.receive()
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            if msg_type != "websocket.receive":
                continue

            raw_bytes = msg.get("bytes")
            raw_text = msg.get("text")

            if raw_bytes:
                if not state["paused"]:
                    sr = state["sample_rate"]
                    if sr == 16000:
                        recorder.feed_audio(raw_bytes)          # 16k int16 bytes 直送
                    else:
                        arr = np.frombuffer(raw_bytes, dtype=np.int16)
                        recorder.feed_audio(arr, sr)            # numpy 経路でリサンプル
            elif raw_text:
                try:
                    ctrl = json.loads(raw_text)
                    t = ctrl.get("type")
                    if t == "config":
                        state["sample_rate"] = int(ctrl.get("sampleRate", 16000))
                        state["mode"] = ctrl.get("mode", "chat")
                    elif t == "pause":
                        state["paused"] = True
                    elif t == "resume":
                        state["paused"] = False
                        recorder.clear_audio_queue()            # 残留音声を破棄
                    elif t == "generate":
                        # クライアントが送ってきた表示中会話コンテキスト
                        gen_q.put_nowait(ctrl.get("messages", []))
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        state["connected"] = False
        active["loop"] = None
        active["queue"] = None
        try:
            recorder.abort()                                    # recorder.text() のブロック解除
        except Exception:
            pass
        gen_q.put_nowait(None)                                  # worker の gen_q.get() を解除
        out_q.put_nowait(None)                                  # sender タスクを終了
        await sender_task


@app.post("/api/summary")
async def summary(req: Request):
    body = await req.json()
    utterances = [u for u in body.get("utterances", []) if u and u.strip()]
    text = summarize_utterances(llm_client, utterances)
    return JSONResponse({"summary": text})


@app.post("/api/reset")
def reset():
    # サーバ側に永続履歴は無いので no-op。画面クリアはクライアント側で行う。
    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
