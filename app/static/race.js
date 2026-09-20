const SAMPLE_RATE = 16000;
const FRAME = 320;
const $ = id => document.getElementById(id);

const lanes = {
  baseline: {
    root: $('baselineLane'),
    transcript: $('baselineTranscript'),
    state: $('baselineState'),
    first: $('baselineFirst'),
    final: $('baselineFinal'),
    confidence: $('baselineConfidence'),
    socket: null,
    sawPartial: false,
    beganAt: 0,
  },
  tuned: {
    root: $('tunedLane'),
    transcript: $('tunedTranscript'),
    state: $('tunedState'),
    first: $('tunedFirst'),
    final: $('tunedFinal'),
    confidence: $('tunedConfidence'),
    socket: null,
    sawPartial: false,
    beganAt: 0,
  },
};

const fixture = {
  baseline: [
    { at: 260, text: 'who', stable: 0, confidence: .58 },
    { at: 590, text: 'who were', stable: 3, confidence: .49 },
    { at: 930, text: 'who were killed', stable: 8, confidence: .43 },
    { at: 1240, final: true, text: 'who were killed', confidence: .43 },
  ],
  tuned: [
    { at: 310, text: 'wick', stable: 0, confidence: .71 },
    { at: 680, text: 'wicked', stable: 4, confidence: .79 },
    { at: 960, final: true, text: 'wicked', confidence: .815 },
  ],
};

let sourceFile = null;
let sequence = 0;
let fixtureTimers = [];
let toastTimer = null;

function renderTranscript(lane, text, stableLength = text.length) {
  const cut = Math.max(0, Math.min(stableLength, text.length));
  lane.transcript.textContent = text.slice(0, cut);
  if (cut < text.length) {
    const tail = document.createElement('span');
    tail.className = 'tail';
    tail.textContent = text.slice(cut);
    lane.transcript.append(tail);
  }
}

function setLaneState(lane, label, className = '') {
  lane.state.textContent = label;
  lane.root.classList.remove('running', 'complete', 'error');
  if (className) lane.root.classList.add(className);
}

function resetLane(lane) {
  lane.socket?.close();
  lane.socket = null;
  lane.sawPartial = false;
  lane.transcript.innerHTML = '<span class="empty">Waiting for the same clip.</span>';
  lane.first.textContent = '-';
  lane.final.textContent = '-';
  lane.confidence.textContent = '-';
  setLaneState(lane, 'READY');
}

function resetRace() {
  fixtureTimers.forEach(timer => clearTimeout(timer));
  fixtureTimers = [];
  sequence = 0;
  resetLane(lanes.baseline);
  resetLane(lanes.tuned);
}

function showToast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.add('show');
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 2200);
}

function handleMessage(lane, message) {
  if (message.type === 'partial') {
    if (!lane.sawPartial) {
      lane.sawPartial = true;
      lane.first.textContent = `${Math.round(performance.now() - lane.beganAt)} ms`;
    }
    renderTranscript(lane, message.text, message.stable_prefix_len);
    lane.confidence.textContent = `${Math.round((message.confidence ?? 0) * 100)}%`;
    return;
  }
  if (message.type === 'final') {
    renderTranscript(lane, message.text);
    lane.final.textContent = `${Math.round(performance.now() - lane.beganAt)} ms`;
    lane.confidence.textContent = `${Math.round((message.confidence ?? 0) * 100)}%`;
    setLaneState(lane, 'COMPLETE', 'complete');
    return;
  }
  if (message.type === 'error') setLaneState(lane, 'MODEL ERROR', 'error');
}

function connectLane(lane, url) {
  return new Promise((resolve, reject) => {
    lane.socket = new WebSocket(url);
    lane.socket.onopen = resolve;
    lane.socket.onerror = () => {
      setLaneState(lane, 'CONNECTION FAILED', 'error');
      reject(new Error('WebSocket connection failed'));
    };
    lane.socket.onmessage = event => handleMessage(lane, JSON.parse(event.data));
    lane.socket.onclose = () => {
      if (!lane.root.classList.contains('complete') && !lane.root.classList.contains('error')) {
        setLaneState(lane, 'CLOSED');
      }
    };
  });
}

function floatToBase64(floatSamples) {
  const pcm = new Int16Array(floatSamples.length);
  for (let index = 0; index < floatSamples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, floatSamples[index]));
    pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function sendSharedFrame(floatSamples) {
  const message = JSON.stringify({ type: 'audio', seq: sequence, pcm16_b64: floatToBase64(floatSamples) });
  lanes.baseline.socket.send(message);
  lanes.tuned.socket.send(message);
  sequence += 1;
}

async function runLiveRace() {
  if (!sourceFile) {
    toggleSetup(true);
    showToast('CHOOSE ONE SOURCE WAV FIRST');
    return;
  }
  resetRace();
  setLaneState(lanes.baseline, 'CONNECTING', 'running');
  setLaneState(lanes.tuned, 'CONNECTING', 'running');
  try {
    await Promise.all([
      connectLane(lanes.baseline, $('baselineUrl').value.trim()),
      connectLane(lanes.tuned, $('tunedUrl').value.trim()),
    ]);
    const beganAt = performance.now();
    Object.values(lanes).forEach(lane => {
      lane.beganAt = beganAt;
      setLaneState(lane, 'LISTENING', 'running');
    });
    const context = new AudioContext({ sampleRate: SAMPLE_RATE });
    const buffer = await context.decodeAudioData(await sourceFile.arrayBuffer());
    const playback = context.createBufferSource();
    playback.buffer = buffer;
    playback.connect(context.destination);
    playback.start();
    const samples = buffer.getChannelData(0);
    for (let offset = 0; offset < samples.length; offset += FRAME) {
      sendSharedFrame(samples.subarray(offset, offset + FRAME));
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const end = JSON.stringify({ type: 'end' });
    lanes.baseline.socket.send(end);
    lanes.tuned.socket.send(end);
    playback.onended = () => context.close();
  } catch (error) {
    Object.values(lanes).forEach(lane => {
      try { lane.socket?.close(); } catch {}
      lane.socket = null;
      setLaneState(lane, 'CONNECTION FAILED', 'error');
    });
    showToast(error.message.toUpperCase());
  }
}

function runFixtureLane(lane, messages) {
  lane.beganAt = performance.now();
  setLaneState(lane, 'LISTENING', 'running');
  messages.forEach(message => {
    fixtureTimers.push(setTimeout(() => {
      handleMessage(lane, message.final
        ? { type: 'final', text: message.text, confidence: message.confidence }
        : { type: 'partial', text: message.text, stable_prefix_len: message.stable, confidence: message.confidence });
    }, message.at));
  });
}

function runFixtureRace() {
  resetRace();
  $('sourceName').textContent = 'M02 · 1_M02_2_headMic_0233.wav';
  runFixtureLane(lanes.baseline, fixture.baseline);
  runFixtureLane(lanes.tuned, fixture.tuned);
}

function toggleSetup(force) {
  const open = force ?? !$('setupDrawer').classList.contains('open');
  $('setupDrawer').classList.toggle('open', open);
  $('setupDrawer').setAttribute('aria-hidden', String(!open));
  if (open) $('baselineUrl').focus();
}

function handleKeyboard(event) {
  if (event.key.toLowerCase() === 'c') {
    event.preventDefault();
    toggleSetup();
    return;
  }
  if ($('setupDrawer').classList.contains('open')) {
    if (event.key === 'Escape') toggleSetup(false);
    return;
  }
  if (['INPUT', 'BUTTON'].includes(event.target.tagName)) return;
  if (event.key === ' ') {
    event.preventDefault();
    runFixtureRace();
  } else if (event.key.toLowerCase() === 'l') {
    runLiveRace();
  } else if (event.key.toLowerCase() === 'r') {
    resetRace();
  }
}

$('fixtureRun').onclick = runFixtureRace;
$('liveRun').onclick = runLiveRace;
$('resetRace').onclick = resetRace;
$('closeSetup').onclick = () => toggleSetup(false);
$('sourceFile').onchange = event => {
  sourceFile = event.target.files[0] || null;
  $('sourceFileName').textContent = sourceFile?.name || 'Choose one local WAV';
  $('sourceName').textContent = sourceFile?.name.toUpperCase() || 'BUILT-IN M02 FIXTURE';
};
document.addEventListener('keydown', handleKeyboard);
