from faster_whisper import WhisperModel

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    "audio.wav",
    language="ja",
    beam_size=5,
    vad_filter=True
)

for seg in segments:
    print(seg.text)
