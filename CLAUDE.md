# Talkie-Duck — コードベース概要

日本語音声チャット Web アプリ。Python(FastAPI) バックエンド + バニラ JS フロントエンド。ビルドステップなし、フレームワークなし。

## ファイル構成

```
server.py          FastAPI サーバ。WebSocket・REST エンドポイント
core.py            STT/LLM/TTS のラッパ関数群、設定定数
main.py            CLI 版（サーバ側マイク使用）。Web UI とは独立
static/
  index.html       UI 全体。ヘッダー・チャット・フッターのみ
  app.js           フロントエンドロジック全体（バニラ JS、IIFE）
  style.css        スタイル
  pcm-worklet.js   AudioWorklet: Float32 → int16 変換してメインスレッドへ
device check.py    マイクデバイス番号確認用スクリプト
run-voicevox-cpu.sh  VoiceVox Docker 起動スクリプト
```

## 重要な設計方針

**DOM が会話コンテキストの唯一の真実。**

サーバは会話履歴を永続化しない。ページをリロードすれば履歴はゼロになる。各発話確定時にブラウザが表示中の全バブルを `generate` メッセージで送り、サーバはその場限りの LLM 呼び出しに使う。要約も同様に、ブラウザが `.bubble.user` テキストを収集して `/api/summary` に POST する。

この設計の理由: サーバ永続履歴は (1) リロードしても残る、(2) `max_turns` でトリミングされる、(3) モード混在が起きる、の 3 点で画面表示とズレていたため。

## ターンフロー（会話モード）

```
[ブラウザ]                         [サーバ worker スレッド]
  PCM バイナリ送信 (連続)  →  recorder.feed_audio()
                                   recorder.text() でブロック
  ← transcript_interim           _on_realtime コールバック
  ← transcript_final             recorder.text() 返却
  pause + generate(messages) →   gen_q.put_nowait(messages)
                                   gen_q.get() で待機解除
                                   chat_stream() で LLM ストリーム
  ← reply_delta (逐次)
  ← audio (文単位 WAV)            voicevox_synthesize() per sentence
  ← done
  resume →                        recorder.clear_audio_queue()
```

**書き起こしモード**では `generate` は送らず、サーバは `transcript_final` 後に `continue` するだけ。LLM/TTS は呼ばれない。

## WebSocket メッセージ一覧

### クライアント → サーバ

| type | 内容 |
|------|------|
| バイナリ | int16 PCM チャンク (16kHz) |
| `config` | `{ sampleRate, mode }` — 接続直後に1回送る |
| `pause` | LLM 処理中のマイク停止 |
| `resume` | 再生完了後のマイク再開 |
| `generate` | `{ messages: [{role,content},...] }` — 表示中全バブル |

### サーバ → クライアント

| type | 内容 |
|------|------|
| `transcript_interim` | `{ text }` — 暫定テキスト |
| `transcript_final` | `{ text }` — 確定テキスト |
| `reply_delta` | `{ text }` — LLM デルタ |
| `audio` | `{ b64 }` — base64 WAV |
| `done` | ストリーム完了 |
| `reset` | 音声コマンドリセット（画面クリア指示） |
| `error` | `{ message }` |

## core.py の主要関数

| 関数 | 用途 |
|------|------|
| `create_recorder(on_realtime, *, use_microphone, input_device_index)` | RealtimeSTT レコーダー生成 |
| `chat_stream(client, messages)` | メッセージ列をそのまま LLM へ（サーバが使う） |
| `chat_completion_stream(client, history, user_text)` | ConversationHistory を使う旧版（CLI `main.py` が使う） |
| `summarize_utterances(client, utterances)` | 発話リスト → Markdown 要約 |
| `summarize_user_utterances(client, history)` | 後方互換ラッパ（CLI 用） |
| `voicevox_synthesize(text, speaker_id)` | VoiceVox TTS → WAV bytes |
| `split_complete_sentences(buf)` | ストリームバッファから完成文を切り出す |

`ConversationHistory` クラスは CLI(`main.py`)専用。Web サーバ側では使わない。

## server.py の並行処理構造

```
asyncio イベントループ (uvicorn)
  ├─ ws_converse (async)
  │    ├─ sender タスク (asyncio): out_q → WebSocket 送信
  │    └─ 受信ループ (async): PCM バイナリ → recorder.feed_audio
  │                           テキスト制御メッセージのルーティング
  └─ worker スレッド (threading.Thread, daemon=True)
       ├─ recorder.text() でブロック
       ├─ gen_q.get() で generate メッセージを待つ
       └─ push() で asyncio キュー(out_q)へスレッドセーフに書込み
```

スレッド間の橋渡し:
- `out_q` (asyncio.Queue): worker → sender タスク
- `gen_q` (queue.Queue): 受信ループ → worker スレッド
- `loop.call_soon_threadsafe()`: _on_realtime コールバック → out_q

## 設定（core.py 先頭）

```python
WHISPER_USE_GPU = True
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "default"
VOICEVOX_BASE_URL = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1
VOICEVOX_SPEED = 1.4
SYSTEM_PROMPT = "..."
```

STT モデル（`create_recorder` 内）: `kotoba-tech/kotoba-whisper-v2.0-faster`
RealtimeSTT パラメータ: `post_speech_silence_duration=1.5`, `min_length_of_recording=0.5`

## 既知の制約

- 単一ユーザー前提。WebSocket 多重接続は `code=1008` で拒否する
- `main.py` のマイクデバイス番号（`AUDIO_INPUT_DEVICE`）は環境依存で手動設定が必要
- VoiceVox ネイティブアプリが起動済みであること（または Docker で `run-voicevox-cpu.sh`）
- LM Studio でモデルをロードしてサーバを起動しておくこと
