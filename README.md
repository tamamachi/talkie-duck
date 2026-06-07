# Talkie-Duck

ローカル LLM と音声で会話したり、話した内容を書き起こして要約したりできる日本語音声チャット Web アプリです。すべてローカルで動作し、外部サービスへの通信は一切ありません。

## 機能

- **AI会話モード** — マイクに向かって話すと、LLM が短く返答し、VoiceVox が音声で読み上げます
- **書き起こしモード** — LLM 返答なしで発話をテキストとして積み上げます。ボタン操作不要で連続して話し続けられます
- **概要をまとめる** — 現在画面に表示されている発言を LLM が Markdown リストに要約します
- リアルタイム音声認識プレビュー（話しながらテキストが更新される）
- 音声コマンド「リセット」で画面をクリア

## 必要なもの

| 依存 | 用途 | 備考 |
|------|------|------|
| Python 3.12 | サーバ実行 | |
| CUDA 対応 GPU | Whisper の高速推論 | `WHISPER_USE_GPU = False` で CPU 動作可（低速） |
| [LM Studio](https://lmstudio.ai/) | ローカル LLM | `localhost:1234` でサーバ起動しておく |
| [VoiceVox](https://voicevox.hiroshiba.jp/) | 日本語 TTS | ネイティブアプリまたは Docker |

VoiceVox を Docker で起動する場合（CPU 版）:

```sh
bash run-voicevox-cpu.sh
```

## セットアップ

```sh
python -m venv venv
venv\Scripts\activate        # Windows
# または source venv/bin/activate  (Linux/Mac)

pip install fastapi uvicorn[standard] openai httpx numpy RealtimeSTT
```

> RealtimeSTT は初回起動時に `kotoba-tech/kotoba-whisper-v2.0-faster` モデルをダウンロードします（数 GB）。

## 起動

LM Studio と VoiceVox を起動した状態で:

```sh
python server.py
```

ブラウザで `http://127.0.0.1:8000` を開きます。

## 使い方

1. ヘッダーのトグルでモードを選択（**AI会話** / **書き起こし**）
2. 「会話開始」または「メモ開始」ボタンを押してマイクをオンにする
3. 話す → 認識が確定したらバブルに表示される
4. 「概要をまとめる」を押すと、現在画面に表示されている発言を要約
5. 「リセット」で画面をクリア

## 設定

`core.py` の先頭にある定数で変更できます:

```python
WHISPER_USE_GPU = True            # False にすると CPU 推論
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
VOICEVOX_BASE_URL  = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1          # 話者 ID
VOICEVOX_SPEED = 1.4             # 読み上げ速度
SYSTEM_PROMPT = "..."            # LLM へのシステムプロンプト
```

## CLI 版

Web UI なしでサーバ側マイクを使って動かすこともできます:

```sh
python main.py
```

`main.py` の `AUDIO_INPUT_DEVICE` に使用するマイクのデバイス番号を設定してください（`device_check.py` で確認できます）。

## アーキテクチャ

```
ブラウザ
  ├─ マイク → PCM(16kHz int16) → WebSocket → FastAPI
  │                                               ├─ RealtimeSTT (Whisper)
  │                                               │    暫定/確定テキスト → WS → UI バブル
  │                                               ├─ [AI会話モード]
  │                                               │    LLM (LM Studio) → 文分割 → VoiceVox → WS → 再生
  │                                               └─ [書き起こしモード]
  │                                                    テキストのみ記録、LLM/TTS なし
  └─ 概要ボタン → POST /api/summary (表示中発言を JSON で送信) → LLM → 表示
```

会話コンテキストは**ブラウザの DOM が唯一の真実**です。サーバは会話履歴を永続化しません。ページをリロードすると履歴はリセットされます。
