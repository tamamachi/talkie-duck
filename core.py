import tempfile
import os

import httpx
import openai
from faster_whisper import WhisperModel

# --- 設定 ---
WHISPER_USE_GPU = True

SAMPLE_RATE = 16000

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "default"

VOICEVOX_BASE_URL = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1
VOICEVOX_SPEED = 1.4

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

    def user_utterances(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "user"]


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


# --- STT ---
def transcribe(model: WhisperModel, audio_path: str) -> str:
    segments, _ = model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        vad_filter=True,
    )
    return "".join(seg.text for seg in segments).strip()


# --- LLM ---
def make_llm_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


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


def summarize_user_utterances(client: openai.OpenAI, history: ConversationHistory) -> str:
    utterances = history.user_utterances()
    if not utterances:
        return "まだ発言がありません。"
    numbered = "\n".join(f"{i+1}. {u}" for i, u in enumerate(utterances))
    messages = [
        {
            "role": "system",
            "content": "以下は今回ユーザーが話した発言の一覧です。要点を簡潔な日本語でまとめてください。",
        },
        {"role": "user", "content": numbered},
    ]
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


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
