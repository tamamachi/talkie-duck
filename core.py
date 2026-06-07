from collections.abc import Generator

import httpx
import openai

SENTENCE_ENDINGS = "。．.!?！？\n"

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


# --- STT: RealtimeSTT レコーダー ---
def create_recorder(on_realtime, *, use_microphone: bool = True, input_device_index=None):
    """
    RealtimeSTT の AudioToTextRecorder を生成して返す。
    - on_realtime: 暫定テキスト更新時に呼ばれるコールバック (str) -> None
    - use_microphone=False の場合は feed_audio() で外部音声を投入する想定
    - input_device_index: サーバ側マイク利用時のデバイス番号（None でデフォルト）
    注: デフォルトの realtime_model_type は "tiny"。精度不足なら "small" 等に変更可。
    """
    from RealtimeSTT import AudioToTextRecorder
    return AudioToTextRecorder(
        model="kotoba-tech/kotoba-whisper-v2.0-faster",
        language="ja",
        device="cuda" if WHISPER_USE_GPU else "cpu",
        compute_type="float16" if WHISPER_USE_GPU else "int8",
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_realtime,
        use_microphone=use_microphone,
        input_device_index=input_device_index,
        silero_sensitivity=0.4,
        webrtc_sensitivity=3,
        post_speech_silence_duration=1.5,
        min_length_of_recording=0.5,
        spinner=False,
    )


# --- 文分割ユーティリティ ---
def split_complete_sentences(buf: str) -> tuple[list[str], str]:
    """バッファから完成した文のリストと残りバッファを返す。"""
    sentences = []
    last = 0
    for i, ch in enumerate(buf):
        if ch in SENTENCE_ENDINGS:
            s = buf[last:i + 1].strip()
            if s:
                sentences.append(s)
            last = i + 1
    return sentences, buf[last:]


# --- LLM ---
def make_llm_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


def chat_completion_stream(
    client: openai.OpenAI, history: ConversationHistory, user_text: str
) -> Generator[str, None, None]:
    history.add_user(user_text)
    full_reply = []
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=history.get_messages(),
        temperature=0.7,
        stream=True,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_reply.append(delta)
            yield delta
    history.add_assistant("".join(full_reply))


def chat_stream(
    client: openai.OpenAI, messages: list[dict]
) -> Generator[str, None, None]:
    """システムプロンプト込みの messages をそのまま受け取り LLM をストリーム呼出しする。
    サーバ永続履歴を使わずクライアント提供のコンテキストだけで動作する版。"""
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def summarize_utterances(client: openai.OpenAI, utterances: list[str]) -> str:
    """発話リストを受け取り要点をMarkdownリストで返す。"""
    if not utterances:
        return "まだ発言がありません。"
    numbered = "\n".join(f"{i+1}. {u}" for i, u in enumerate(utterances))
    messages = [
        {
            "role": "system",
            "content": """以下は今回ユーザーが話した発言の一覧です。要点をまとめて、Markdownのリスト形式に変換して。必要ならネストも使ってください。
            注意点として、ユーザは音声入力でテキストを作成しているので、音声認識の誤りが混入している可能性があります。そのような部分は文脈からユーザの発言内容を類推して読み替えてください。""",
        },
        {"role": "user", "content": numbered},
    ]
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def summarize_user_utterances(client: openai.OpenAI, history: ConversationHistory) -> str:
    """後方互換: ConversationHistory からユーザー発言を取り出して要約する。"""
    return summarize_utterances(client, history.user_utterances())


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
