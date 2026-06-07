/**
 * PCM Worklet: Float32 マイク入力を int16 に変換してメインスレッドへ transfer。
 * AudioContext の sampleRate で動作し、約 2048 サンプルごとにバッファを送出する。
 * メインスレッド側は受け取った ArrayBuffer を WebSocket でサーバへ直接送信できる。
 */
class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(0);
    this._CHUNK = 2048; // 送出単位（samples）
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;

    // Float32 → int16（クリップ付き）
    const int16 = new Int16Array(ch.length);
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i] < -1 ? -1 : ch[i] > 1 ? 1 : ch[i];
      int16[i] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7fff) | 0;
    }

    // バッファに追記
    const merged = new Int16Array(this._buf.length + int16.length);
    merged.set(this._buf);
    merged.set(int16, this._buf.length);
    this._buf = merged;

    // CHUNK ごとに transfer で送出（コピーなし）
    while (this._buf.length >= this._CHUNK) {
      const chunk = this._buf.slice(0, this._CHUNK);
      this._buf = this._buf.slice(this._CHUNK);
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PcmProcessor);
