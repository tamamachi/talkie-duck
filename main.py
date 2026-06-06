import io
import os
import queue
import tempfile
import wave
from collections import deque

import httpx
import numpy as np
import openai
import sounddevice as sd
from faster_whisper import WhisperModel

# --- 設定 ---
WHISPER_USE_GPU = True  # True にするとGPU (CUDA) モードで起動

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCK_MS = 30
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000  # 480 samples

SILENCE_DURATION_S = 1.5
MIN_RECORDING_S = 0.5
SILENCE_BLOCKS = int(SILENCE_DURATION_S * 1000 / BLOCK_MS)

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "default"

VOICEVOX_BASE_URL = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1
VOICEVOX_SPEED = 1.4  # 話す速さ (0.5〜2.0、1.0が標準)

AUDIO_INPUT_DEVICE = 3  # ヘッドセット (mutalk 0306411 Hands-Free)

SYSTEM_PROMPT = """親しい友人のようにユーザと雑談してください。ユーザは音声認識されたテキストを入力しているので、誤認識っぽいところはうまく類推して読み替えてください。
返事の内容は、人間が雑談でする返事のように、返答は短く、相槌＋ひとこと　程度にとどめてください。
口調は親しい友人と話すときのようにタメ口で。あなたの返事はTTS読み上げソフトへの入力になるので、記号や絵文字は使わずにそのまま読み上げ可能な文字列で返事をして。"""

# --- 会話履歴 ---
class ConversationHistory:
    def __init__(self, system_prompt: str, max_turns: int = 20):
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self.max_turns = max_turns

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def _trim(self):
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def clear(self):
        self.messages = []


# --- Whisperモデル ---
def create_whisper_model() -> WhisperModel:
    if WHISPER_USE_GPU:
        print("Whisperモデルを読み込んでいます (GPUモード)...")
        return WhisperModel(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cuda",
            compute_type="float16",
        )
    else:
        print("Whisperモデルを読み込んでいます (CPUモード)...")
        return WhisperModel(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cpu",
            compute_type="int8",
        )


# --- VAD ---
def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def calibrate_noise(q: queue.Queue) -> float:
    print("環境音を計測中... (1秒間静かにしてください)")
    blocks = []
    target_blocks = int(1000 / BLOCK_MS)  # 1秒分
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


# --- WAV保存 ---
def save_wav(audio: np.ndarray, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


# --- STT ---
def transcribe(model: WhisperModel, wav_path: str) -> str:
    segments, _ = model.transcribe(
        wav_path,
        language="ja",
        beam_size=5,
        vad_filter=True,
    )
    return "".join(seg.text for seg in segments).strip()


# --- LLM ---
def chat_completion(client: openai.OpenAI, history: ConversationHistory, user_text: str) -> str:
    history.add_user(user_text)
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=history.get_messages(),
        temperature=0.7,
    )
    reply = response.choices[0].message.content
    history.add_assistant(reply)
    return reply


# --- TTS ---
def voicevox_synthesize(text: str, speaker_id: int = VOICEVOX_SPEAKER_ID) -> bytes:
    resp1 = httpx.post(
        f"{VOICEVOX_BASE_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30.0,
    )
    resp1.raise_for_status()
    query = resp1.json()
    query["speedScale"] = VOICEVOX_SPEED

    resp2 = httpx.post(
        f"{VOICEVOX_BASE_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=60.0,
    )
    resp2.raise_for_status()
    return resp2.content


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


# --- メインループ ---
def main_loop():
    model = create_whisper_model()
    history = ConversationHistory(SYSTEM_PROMPT)
    llm_client = openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")
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

            # リセットコマンド
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
