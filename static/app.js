(() => {
  const btnToggle    = document.getElementById('btn-toggle');
  const btnSummary   = document.getElementById('btn-summary');
  const btnReset     = document.getElementById('btn-reset');
  const statusEl     = document.getElementById('status');
  const chatEl       = document.getElementById('chat');
  const summaryPanel = document.getElementById('summary-panel');
  const summaryText  = document.getElementById('summary-text');
  const btnModeChat  = document.getElementById('mode-chat');
  const btnModeMemo  = document.getElementById('mode-memo');

  let mode    = 'chat';   // 'chat' | 'memo'
  let running    = false;
  let paused     = false;   // true = AI 返答の再生中（PCM 送信を停止）
  let audioCtx   = null;
  let mediaStream = null;
  let workletNode = null;
  let ws         = null;

  // 現在ターンのバブル参照
  let currentUserBubble = null;
  let currentAiBubble   = null;
  let currentTurnId     = null;

  // --- 音声再生キュー ---
  const audioQueue = [];
  let audioPlaying = false;
  let streamDone   = false;

  function setStatus(text, cls = '') {
    statusEl.textContent = text;
    statusEl.className = cls;
  }

  // --- 表示中バブルの収集 ---
  function collectMessages() {
    // 会話用: 表示順に role 付きメッセージ列（user/.bubble.user, assistant/.bubble.ai）
    return Array.from(chatEl.querySelectorAll('.bubble'))
      .map(el => ({
        role: el.classList.contains('user') ? 'user' : 'assistant',
        content: el.textContent.trim(),
      }))
      .filter(m => m.content);
  }

  function collectUtterances() {
    // 要約用: 表示中のユーザー発言テキストだけ
    return Array.from(chatEl.querySelectorAll('.bubble.user'))
      .map(el => el.textContent.trim())
      .filter(Boolean);
  }

  function getOrCreateBubble(id, role) {
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement('div');
      el.id = id;
      el.className = `bubble ${role}`;
      chatEl.appendChild(el);
    }
    chatEl.scrollTop = chatEl.scrollHeight;
    return el;
  }

  // --- 音声キュー再生 ---
  function enqueueAudio(b64) {
    audioQueue.push(b64);
    if (!audioPlaying) playNext();
  }

  function playNext() {
    if (audioQueue.length === 0) {
      audioPlaying = false;
      checkDone();
      return;
    }
    audioPlaying = true;
    setStatus('再生中', 'speaking');
    const b64   = audioQueue.shift();
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const url   = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); playNext(); };
    audio.play();
  }

  function checkDone() {
    if (streamDone && !audioPlaying && audioQueue.length === 0) {
      // 再生完了 → マイク送信再開をサーバへ通知
      paused = false;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resume' }));
      }
      if (running) setStatus('待機中');
    }
  }

  // --- WebSocket メッセージ処理 ---
  function handleWsMessage(event) {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    if (data.type === 'transcript_interim') {
      // 暫定テキスト: 新ターン開始またはバブル上書き
      if (!currentTurnId) {
        currentTurnId     = Date.now();
        currentUserBubble = getOrCreateBubble(`user-${currentTurnId}`, 'user');
        currentAiBubble   = null;
        streamDone        = false;
        audioQueue.length = 0;
        audioPlaying      = false;
      }
      if (currentUserBubble) {
        currentUserBubble.textContent = data.text;   // 追記ではなく上書き
        chatEl.scrollTop = chatEl.scrollHeight;
      }
      setStatus('聞いています', 'listening');

    } else if (data.type === 'transcript_final') {
      // 確定テキスト: バブルを確定表示して次のターン用に参照をリセット
      if (!currentTurnId) {
        currentTurnId     = Date.now();
        currentUserBubble = getOrCreateBubble(`user-${currentTurnId}`, 'user');
      }
      if (currentUserBubble) {
        currentUserBubble.textContent = data.text;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
      currentUserBubble = null;   // 次の interim は新バブルを作る

      if (mode === 'memo') {
        currentTurnId = null;   // メモモード: done が来ないのでここでリセット
        // メモモード: LLM待ちなし。マイクを止めず連続発話できる
        setStatus('待機中');
      } else {
        // 会話モード: LLM 問い合わせ中は録音を止める（AI 返答再生終了後に resume）
        paused = true;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'pause' }));
          // 確定済みバブルを含む表示中全コンテキストをサーバへ送信
          ws.send(JSON.stringify({ type: 'generate', messages: collectMessages() }));
        }
        setStatus('考え中...', 'processing');
      }

    } else if (data.type === 'reply_delta') {
      if (!currentTurnId) currentTurnId = Date.now();
      if (!currentAiBubble) {
        currentAiBubble = getOrCreateBubble(`ai-${currentTurnId}`, 'ai');
        setStatus('応答中...', 'processing');
      }
      currentAiBubble.textContent += data.text;
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (data.type === 'audio') {
      enqueueAudio(data.b64);

    } else if (data.type === 'done') {
      streamDone    = true;
      currentTurnId = null;       // 次のターン準備
      currentAiBubble = null;
      checkDone();

    } else if (data.type === 'reset') {
      chatEl.innerHTML = '';

    } else if (data.type === 'error') {
      console.error('[server error]', data.message);
    }
  }

  // --- 会話開始 ---
  async function startConversation() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch {
      alert('マイクのアクセスが許可されていません。');
      return;
    }

    // 16kHz AudioContext（サーバの RealtimeSTT に合わせる）
    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.resume();

    try {
      await audioCtx.audioWorklet.addModule('/pcm-worklet.js');
    } catch (e) {
      console.error('AudioWorklet ロード失敗:', e);
      alert('AudioWorklet の読み込みに失敗しました。');
      return;
    }

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode  = new AudioWorkletNode(audioCtx, 'pcm-processor');
    source.connect(workletNode);
    // playback 不要なので destination には繋がない

    // WebSocket 接続
    ws = new WebSocket(`ws://${location.host}/ws/converse`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      // 実際の sampleRate とモードをサーバへ通知
      ws.send(JSON.stringify({ type: 'config', sampleRate: audioCtx.sampleRate, mode }));
      setStatus('待機中');
    };
    ws.onmessage = handleWsMessage;
    ws.onclose   = () => { if (running) setStatus('切断'); };
    ws.onerror   = (e) => { console.error('WS エラー:', e); };

    // PCM チャンクを WS へ送出（再生中は停止）
    workletNode.port.onmessage = (e) => {
      if (running && !paused && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(e.data);
      }
    };

    running = true;
    paused  = false;
    btnToggle.textContent = mode === 'memo' ? 'メモ停止' : '会話停止';
    btnToggle.classList.add('active');
    btnModeChat.disabled = true;
    btnModeMemo.disabled = true;
    setStatus('接続中...');
  }

  // --- 会話停止 ---
  function stopConversation() {
    running = false;
    paused  = false;

    if (ws)          { ws.close(); ws = null; }
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (audioCtx)    { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

    currentUserBubble = null;
    currentAiBubble   = null;
    currentTurnId     = null;

    btnToggle.textContent = mode === 'memo' ? 'メモ開始' : '会話開始';
    btnToggle.classList.remove('active');
    btnModeChat.disabled = false;
    btnModeMemo.disabled = false;
    setStatus('待機中');
  }

  // --- ボタン ---
  btnToggle.addEventListener('click', () => {
    if (!running) startConversation(); else stopConversation();
  });

  function setMode(m) {
    mode = m;
    btnModeChat.classList.toggle('active', m === 'chat');
    btnModeMemo.classList.toggle('active', m === 'memo');
    if (!running) {
      btnToggle.textContent = m === 'memo' ? 'メモ開始' : '会話開始';
    }
  }
  btnModeChat.addEventListener('click', () => setMode('chat'));
  btnModeMemo.addEventListener('click', () => setMode('memo'));

  btnSummary.addEventListener('click', async () => {
    btnSummary.disabled = true;
    try {
      const utterances = collectUtterances();
      const res  = await fetch('/api/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ utterances }),
      });
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
