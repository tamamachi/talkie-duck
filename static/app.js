(() => {
  const btnToggle    = document.getElementById('btn-toggle');
  const btnSummary   = document.getElementById('btn-summary');
  const btnReset     = document.getElementById('btn-reset');
  const statusEl     = document.getElementById('status');
  const chatEl       = document.getElementById('chat');
  const summaryPanel = document.getElementById('summary-panel');
  const summaryText  = document.getElementById('summary-text');

  let running      = false;
  let paused       = false;
  let audioCtx     = null;
  let analyser     = null;
  let dataArray    = null;
  let mediaStream  = null;
  let mediaRecorder = null;
  let recordChunks  = [];
  let threshold    = 0.015;
  let isRecording  = false;
  let silenceTimer = null;
  let vadTimer     = null;
  let recordStart  = 0;

  const SILENCE_MS    = 1500;
  const MIN_RECORD_MS = 500;

  function setStatus(text, cls = '') {
    statusEl.textContent = text;
    statusEl.className = cls;
  }

  function addBubble(text, role) {
    const div = document.createElement('div');
    div.className = `bubble ${role}`;
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function getRMS() {
    analyser.getFloatTimeDomainData(dataArray);
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i] * dataArray[i];
    return Math.sqrt(sum / dataArray.length);
  }

  async function calibrate() {
    setStatus('環境音を計測中...');
    const samples = [];
    await new Promise(resolve => {
      const start = Date.now();
      const id = setInterval(() => {
        samples.push(getRMS());
        if (Date.now() - start >= 1000) {
          clearInterval(id);
          resolve();
        }
      }, 30);
    });
    const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
    threshold = Math.max(avg * 3, 0.008);
    console.log('threshold:', threshold);
    setStatus('待機中');
  }

  function startVAD() {
    vadTimer = setInterval(() => {
      if (!running || paused) return;

      const rms = getRMS();

      if (!isRecording) {
        if (rms > threshold) startRecording();
      } else {
        if (rms < threshold) {
          if (!silenceTimer) {
            silenceTimer = setTimeout(() => {
              const duration = Date.now() - recordStart;
              if (duration >= MIN_RECORD_MS) stopRecording();
              else cancelRecording();
            }, SILENCE_MS);
          }
        } else {
          clearTimeout(silenceTimer);
          silenceTimer = null;
        }
      }
    }, 30);
  }

  function startRecording() {
    recordChunks = [];
    recordStart  = Date.now();
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordChunks.push(e.data); };
    mediaRecorder.onstop = onRecordStop;
    mediaRecorder.start();
    isRecording = true;
    setStatus('聞いています', 'listening');
  }

  function stopRecording() {
    clearTimeout(silenceTimer);
    silenceTimer = null;
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
  }

  function cancelRecording() {
    clearTimeout(silenceTimer);
    silenceTimer = null;
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
    recordChunks = [];
    setStatus('待機中');
  }

  async function onRecordStop() {
    if (recordChunks.length === 0) { setStatus('待機中'); return; }

    const mimeType = mediaRecorder.mimeType || 'audio/webm';
    const blob = new Blob(recordChunks, { type: mimeType });
    recordChunks = [];

    paused = true;
    setStatus('認識中...', 'processing');

    const ext  = mimeType.includes('ogg') ? '.ogg' : '.webm';
    const form = new FormData();
    form.append('audio', blob, `audio${ext}`);

    let data;
    try {
      const res = await fetch('/api/converse', { method: 'POST', body: form });
      data = await res.json();
    } catch (err) {
      console.error(err);
      setStatus('エラー');
      paused = false;
      return;
    }

    if (!data.transcript) {
      setStatus('待機中');
      paused = false;
      return;
    }

    addBubble(data.transcript, 'user');
    addBubble(data.reply, 'ai');

    if (data.audio_b64) {
      setStatus('再生中', 'speaking');
      const bytes = Uint8Array.from(atob(data.audio_b64), c => c.charCodeAt(0));
      const url   = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
      const audio = new Audio(url);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        paused = false;
        if (running) setStatus('待機中');
      };
      audio.play();
    } else {
      paused = false;
      if (running) setStatus('待機中');
    }
  }

  async function startConversation() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch {
      alert('マイクのアクセスが許可されていません。');
      return;
    }

    audioCtx  = new AudioContext();
    await audioCtx.resume();

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    dataArray = new Float32Array(analyser.fftSize);

    const src = audioCtx.createMediaStreamSource(mediaStream);
    src.connect(analyser);

    running = true;
    paused  = false;
    btnToggle.textContent = '会話停止';
    btnToggle.classList.add('active');

    await calibrate();
    startVAD();
  }

  function stopConversation() {
    running = false;
    paused  = false;
    clearInterval(vadTimer);
    vadTimer = null;
    if (isRecording) cancelRecording();
    if (audioCtx)    { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    analyser  = null;
    dataArray = null;
    btnToggle.textContent = '会話開始';
    btnToggle.classList.remove('active');
    setStatus('待機中');
  }

  btnToggle.addEventListener('click', () => {
    if (!running) startConversation();
    else stopConversation();
  });

  btnSummary.addEventListener('click', async () => {
    btnSummary.disabled = true;
    try {
      const res  = await fetch('/api/summary', { method: 'POST' });
      const data = await res.json();
      summaryText.textContent = data.summary;
      summaryPanel.classList.remove('hidden');
    } finally {
      btnSummary.disabled = false;
    }
  });

  btnReset.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    chatEl.innerHTML = '';
    summaryPanel.classList.add('hidden');
    summaryText.textContent = '';
  });
})();
