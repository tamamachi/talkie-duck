import io
import queue
import threading
import wave

import numpy as np
import sounddevice as sd

from core import (
    SYSTEM_PROMPT,
    ConversationHistory,
    create_recorder,
    make_llm_client,
    chat_completion_stream,
    voicevox_synthesize,
    split_complete_sentences,
)

# --- CLIの設定 ---
AUDIO_INPUT_DEVICE = 1  # ヘッドセット (mutalk 0306411 Hands-Free)

_STOP_SENTINEL = None


# --- 音声再生 ---
def play_audio(wav_bytes: bytes):
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch == 2:
        audio = audio.reshape(-1, 2)
    sd.play(audio, samplerate=sr)
    sd.wait()


def playback_worker(play_q: queue.Queue):
    """別スレッドで音声キューを順次再生する。"""
    while True:
        item = play_q.get()
        if item is _STOP_SENTINEL:
            play_q.task_done()
            break
        try:
            play_audio(item)
        except Exception as e:
            print(f"[再生エラー] {e}")
        play_q.task_done()


# --- メインループ ---
def main_loop():
    history = ConversationHistory(SYSTEM_PROMPT)
    llm_client = make_llm_client()

    def on_realtime(text: str):
        print(f"\r聞いています: {text}    ", end="", flush=True)

    recorder = create_recorder(
        on_realtime,
        use_microphone=True,
        input_device_index=AUDIO_INPUT_DEVICE,
    )

    print("準備完了。話しかけてください。(Ctrl+C で終了)\n")
    try:
        while True:
            # 話し終わるまでブロック。その間 on_realtime が随時呼ばれる
            text = recorder.text()
            if not text or not text.strip():
                continue

            print(f"\rあなた: {text}    ")

            if text.strip() in ("リセット", "最初から", "リセットして"):
                history.clear()
                print("会話履歴をリセットしました。")
                continue

            # 再生中にマイクで拾わないよう録音を一時停止
            recorder.stop()

            print("AI: ", end="", flush=True)
            play_q: queue.Queue = queue.Queue()
            worker = threading.Thread(target=playback_worker, args=(play_q,), daemon=True)
            worker.start()

            try:
                buf = ""
                for delta in chat_completion_stream(llm_client, history, text):
                    print(delta, end="", flush=True)
                    buf += delta
                    sentences, buf = split_complete_sentences(buf)
                    for sentence in sentences:
                        try:
                            play_q.put(voicevox_synthesize(sentence))
                        except Exception as e:
                            print(f"\n[TTS エラー] {e}", end="", flush=True)

                # 残りバッファを合成
                buf = buf.strip()
                if buf:
                    try:
                        play_q.put(voicevox_synthesize(buf))
                    except Exception as e:
                        print(f"\n[TTS エラー] {e}", end="", flush=True)

            except Exception as e:
                print(f"\n[LLM エラー] {e}")

            print()  # 改行

            # 再生完了を待ってからマイクを再開
            play_q.put(_STOP_SENTINEL)
            play_q.join()
            recorder.start()

    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        recorder.abort()


if __name__ == "__main__":
    main_loop()
