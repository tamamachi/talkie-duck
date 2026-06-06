import io
import os
import queue
import tempfile
import wave
from collections import deque

import numpy as np
import sounddevice as sd

from core import (
    SAMPLE_RATE,
    SYSTEM_PROMPT,
    ConversationHistory,
    create_whisper_model,
    make_llm_client,
    chat_completion,
    transcribe,
    voicevox_synthesize,
)

# --- CLIの設定 ---
CHANNELS = 1
DTYPE = "int16"
BLOCK_MS = 30
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000  # 480 samples

SILENCE_DURATION_S = 1.5
MIN_RECORDING_S = 0.5
SILENCE_BLOCKS = int(SILENCE_DURATION_S * 1000 / BLOCK_MS)

AUDIO_INPUT_DEVICE = 3  # ヘッドセット (mutalk 0306411 Hands-Free)


# --- VAD ---
def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def calibrate_noise(q: queue.Queue) -> float:
    print("環境音を計測中... (1秒間静かにしてください)")
    blocks = []
    target_blocks = int(1000 / BLOCK_MS)
    try:
        for _ in range(target_blocks):
            blocks.append(q.get(timeout=5.0).flatten())
    except queue.Empty:
        print("キャリブレーションがタイムアウトしました。デフォルト閾値を使用します。")
        return 500.0
    audio = np.concatenate(blocks)
    ambient = rms(audio)
    threshold = max(ambient * 3.0, 300.0)
    print(f"ノイズ閾値: {threshold:.0f}")
    return threshold


def record_until_silence(q: queue.Queue, threshold: float) -> np.ndarray:
    pre_buf: deque = deque(maxlen=3)
    recording: list[np.ndarray] = []
    silent_blocks = 0
    speech_started = False

    while True:
        block = q.get().flatten()
        level = rms(block)

        if not speech_started:
            pre_buf.append(block)
            if level > threshold:
                speech_started = True
                recording.extend(pre_buf)
                pre_buf.clear()
        else:
            recording.append(block)
            if level <= threshold:
                silent_blocks += 1
                if silent_blocks >= SILENCE_BLOCKS:
                    break
            else:
                silent_blocks = 0

    return np.concatenate(recording) if recording else np.array([], dtype=np.int16)


def save_wav(audio: np.ndarray, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


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


# --- メインループ ---
def main_loop():
    model = create_whisper_model()
    history = ConversationHistory(SYSTEM_PROMPT)
    llm_client = make_llm_client()
    tmp_wav = os.path.join(tempfile.gettempdir(), "talkie_input.wav")

    audio_q: queue.Queue = queue.Queue()

    def mic_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    print("マイクを開いています...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
        device=AUDIO_INPUT_DEVICE,
        callback=mic_callback,
    )
    stream.start()

    threshold = calibrate_noise(audio_q)
    print("\n準備完了。話しかけてください。(Ctrl+C で終了)\n")

    try:
        while True:
            print("聞いています...")
            try:
                audio = record_until_silence(audio_q, threshold)
            except Exception as e:
                print(f"[録音エラー] {e}")
                continue

            if len(audio) < SAMPLE_RATE * MIN_RECORDING_S:
                continue

            try:
                save_wav(audio, tmp_wav)
                text = transcribe(model, tmp_wav)
            except Exception as e:
                print(f"[認識エラー] {e}")
                continue

            if not text:
                print("(認識できませんでした)")
                continue

            if text.strip() in ("リセット", "最初から", "リセットして"):
                history.clear()
                print("会話履歴をリセットしました。")
                continue

            print(f"あなた: {text}")

            try:
                reply = chat_completion(llm_client, history, text)
            except Exception as e:
                print(f"[LLMエラー] {e}")
                continue

            print(f"AI: {reply}")

            try:
                wav_bytes = voicevox_synthesize(reply)
                play_audio(wav_bytes)
            except Exception as e:
                print(f"[音声合成エラー] {e}")

    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main_loop()
