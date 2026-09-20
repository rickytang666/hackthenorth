const SAMPLE_RATE = 16000;
const FRAME = 320;
const CLARIFY_BELOW = 0.70;

const cueCards = [
  { said: 'wicked', baseline: 'who were killed', file: '1_M02_2_headMic_0233.wav', duration: '3.5 sec' },
  { said: 'jagged', baseline: 'they are good', file: '2_M02_1_headMic_0112.wav', duration: '3.3 sec' },
  { said: 'brawn', baseline: 'oh put it on', file: '3_M02_2_headMic_0132.wav', duration: '3.2 sec' },
  { said: 'witty', baseline: 'where is he ah where is he', file: '4_M02_2_headMic_0116.wav', duration: '2.9 sec' },
];

const $ = id => document.getElementById(id);
const scenes = [...document.querySelectorAll('.scene')];
let sceneIndex = 0;
let sourceFile = null;
let voiceFile = null;
let socket = null;
let audioContext = null;
let processor = null;
let mediaStream = null;
let sequence = 0;
let beganAt = 0;
let sawPartial = false;
let finalText = cueCards[0].said;
let finalCandidates = [];
let toastTimer = null;

function buildProgress() {
  $('progressTrack').replaceChildren(...scenes.map(() => document.createElement('i')));
}

function setScene(nextIndex) {
  sceneIndex = Math.max(0, Math.min(scenes.length - 1, nextIndex));
  scenes.forEach((scene, index) => scene.classList.toggle('active', index === sceneIndex));
  $('beatNumber').textContent = String(sceneIndex + 1).padStart(2, '0');
  [...$('progressTrack').children].forEach((bar, index) => {
    bar.className = index < sceneIndex ? 'complete' : index === sceneIndex ? 'current' : '';
  });
  if (sceneIndex === 3) renderClarification();
}

function showToast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.add('show');
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 2200);
}

function setConnection(label, error = false) {
  $('connectionLabel').textContent = label;
  $('connectionPill').classList.toggle('error', error);
}

function renderTranscript(text, stableLength = text.length) {
  const transcript = $('liveTranscript');
  if (!text) {
    transcript.innerHTML = '<span class="empty">Waiting for audio.</span>';
    return;
  }
  const cut = Math.max(0, Math.min(stableLength, text.length));
  transcript.textContent = text.slice(0, cut);
  if (cut < text.length) {
    const tail = document.createElement('span');
    tail.className = 'tail';
    tail.textContent = text.slice(cut);
    transcript.append(tail);
  }
}

function setConfidence(value) {
  const normalized = Math.max(0, Math.min(1, value ?? 0));
  $('confidenceFill').style.width = `${Math.round(normalized * 100)}%`;
  $('confidenceFill').style.background = normalized < CLARIFY_BELOW ? 'var(--signal)' : 'var(--acid)';
  $('confidenceValue').textContent = `${Math.round(normalized * 100)}%`;
}

function applyCue(index) {
  const cue = cueCards[index];
  finalText = cue.said;
  document.querySelectorAll('[data-baseline]').forEach(node => { node.textContent = cue.baseline; });
  document.querySelectorAll('[data-tuned]').forEach(node => { node.textContent = cue.said; });
  $('sourceDuration').textContent = cue.duration.toUpperCase();
}

function loadCueOptions() {
  cueCards.forEach((cue, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${index + 1}. ${cue.said} / ${cue.file}`;
    $('cueSelect').append(option);
  });
  $('cueSelect').onchange = event => applyCue(Number(event.target.value));
}

function toggleSetup(force) {
  const shouldOpen = force ?? !$('setupDrawer').classList.contains('open');
  $('setupDrawer').classList.toggle('open', shouldOpen);
  $('setupDrawer').setAttribute('aria-hidden', String(!shouldOpen));
  if (shouldOpen) $('cueSelect').focus();
}

async function playLocalFile(file, button) {
  if (!file) {
    toggleSetup(true);
    showToast('CHOOSE A LOCAL WAV IN SETUP');
    return;
  }
  if (button) button.classList.add('playing');
  const audio = new Audio(URL.createObjectURL(file));
  audio.onended = () => {
    URL.revokeObjectURL(audio.src);
    if (button) button.classList.remove('playing');
  };
  await audio.play();
}

function connect() {
  return new Promise((resolve, reject) => {
    socket = new WebSocket($('asrUrl').value.trim());
    socket.onopen = () => {
      setConnection('ASR CONNECTED');
      resolve();
    };
    socket.onerror = () => {
      setConnection('ASR UNAVAILABLE', true);
      reject(new Error('WebSocket connection failed'));
    };
    socket.onclose = () => {
      if (!sawPartial) setConnection('LOCAL READY');
    };
    socket.onmessage = event => handleProtocolMessage(JSON.parse(event.data));
  });
}

function handleProtocolMessage(message) {
  if (message.type === 'partial') {
    if (!sawPartial) {
      sawPartial = true;
      $('firstPartial').textContent = `${Math.round(performance.now() - beganAt)} ms`;
    }
    renderTranscript(message.text, message.stable_prefix_len);
    setConfidence(message.confidence);
    return;
  }
  if (message.type === 'final') {
    finalText = message.text;
    finalCandidates = message.candidates || [];
    renderTranscript(finalText);
    setConfidence(message.confidence);
    document.querySelectorAll('[data-tuned]').forEach(node => { node.textContent = finalText; });
    stopCapture();
    setConnection('RECOVERY COMPLETE');
    if ((message.confidence ?? 1) < CLARIFY_BELOW && finalCandidates.length > 1) {
      $('clarifyConfidence').textContent = `${Math.round(message.confidence * 100)}% CONFIDENCE`;
    }
    return;
  }
  if (message.type === 'error') {
    setConnection('MODEL ERROR', true);
    showToast(message.message || 'MODEL ERROR');
  }
}

function sendPcm(floatSamples) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const pcm = new Int16Array(floatSamples.length);
  for (let index = 0; index < floatSamples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, floatSamples[index]));
    pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  socket.send(JSON.stringify({ type: 'audio', seq: sequence, pcm16_b64: btoa(binary) }));
  sequence += 1;
}

function resetRecovery() {
  sequence = 0;
  beganAt = performance.now();
  sawPartial = false;
  finalCandidates = [];
  renderTranscript('');
  setConfidence(0);
  $('firstPartial').textContent = '-';
}

async function replayIntoModel() {
  if (!sourceFile) {
    toggleSetup(true);
    showToast('CHOOSE THE M02 SOURCE WAV FIRST');
    return;
  }
  resetRecovery();
  try {
    await connect();
    setConnection(`REPLAYING ${sourceFile.name.toUpperCase()}`);
    const context = new AudioContext({ sampleRate: SAMPLE_RATE });
    const decoded = await context.decodeAudioData(await sourceFile.arrayBuffer());
    const samples = decoded.getChannelData(0);
    for (let offset = 0; offset < samples.length; offset += FRAME) {
      sendPcm(samples.subarray(offset, offset + FRAME));
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    socket.send(JSON.stringify({ type: 'end' }));
    await context.close();
  } catch (error) {
    setConnection('ASR UNAVAILABLE', true);
    showToast(error.message.toUpperCase());
  }
}

async function startMicrophone() {
  resetRecovery();
  try {
    await connect();
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: SAMPLE_RATE } });
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = event => {
      const samples = event.inputBuffer.getChannelData(0);
      for (let offset = 0; offset < samples.length; offset += FRAME) sendPcm(samples.subarray(offset, offset + FRAME));
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
    $('useMic').textContent = 'STOP MICROPHONE';
    $('useMic').onclick = finishMicrophone;
  } catch (error) {
    setConnection('MICROPHONE UNAVAILABLE', true);
    showToast('MICROPHONE PERMISSION OR ASR FAILED');
  }
}

function finishMicrophone() {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'end' }));
  stopCapture();
}

function stopCapture() {
  try { processor?.disconnect(); } catch {}
  try { audioContext?.close(); } catch {}
  mediaStream?.getTracks().forEach(track => track.stop());
  processor = null;
  audioContext = null;
  mediaStream = null;
  $('useMic').textContent = 'USE MICROPHONE';
  $('useMic').onclick = startMicrophone;
}

function clarificationOptions() {
  if (finalCandidates.length > 1) return finalCandidates.slice(0, 2).map(candidate => candidate.text);
  return ['Do not give her the insulin.', 'Do give her the insulin.'];
}

function renderClarification() {
  const choices = clarificationOptions();
  $('clarifyChoices').replaceChildren(...choices.map((text, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.key = String(index + 1);
    button.textContent = text;
    button.onclick = () => confirmChoice(text, button);
    return button;
  }));
}

function confirmChoice(text, button) {
  finalText = text;
  $('confirmedText').textContent = text.replace(/[.!?]+$/, '');
  document.querySelectorAll('.clarify-choices button').forEach(choice => choice.classList.toggle('selected', choice === button));
  showToast('MEANING CONFIRMED');
  setTimeout(() => setScene(4), 380);
}

async function speakConfirmed() {
  if (voiceFile) {
    await playLocalFile(voiceFile, $('playVoice'));
    $('voiceStatus').textContent = 'PLAYING LOCAL FALLBACK';
    return;
  }
  const base = $('voiceUrl').value.trim().replace(/\/$/, '');
  $('playVoice').classList.add('playing');
  try {
    const response = await fetch(`${base}/v1/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: finalText }),
    });
    if (!response.ok) throw new Error(`Renderer ${response.status}`);
    const context = new AudioContext();
    const buffer = await context.decodeAudioData(await response.arrayBuffer());
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => {
      $('playVoice').classList.remove('playing');
      context.close();
    };
    source.start();
    $('voiceStatus').textContent = 'PERSONAL VOICE LIVE';
  } catch (error) {
    $('playVoice').classList.remove('playing');
    $('voiceStatus').textContent = 'SELECT LOCAL FALLBACK';
    toggleSetup(true);
    showToast(error.message.toUpperCase());
  }
}

function isInteractive(target) {
  return ['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(target.tagName);
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
  if (sceneIndex === 3 && ['1', '2'].includes(event.key)) {
    const choice = $('clarifyChoices').children[Number(event.key) - 1];
    if (choice) choice.click();
    return;
  }
  if (isInteractive(event.target)) return;
  if (['ArrowRight', ' ', 'Enter', 'PageDown'].includes(event.key)) {
    event.preventDefault();
    setScene(sceneIndex + 1);
  } else if (['ArrowLeft', 'Backspace', 'PageUp'].includes(event.key)) {
    event.preventDefault();
    setScene(sceneIndex - 1);
  } else if (event.key === 'Home') {
    setScene(0);
  } else if (event.key === 'End') {
    setScene(scenes.length - 1);
  }
}

buildProgress();
loadCueOptions();
applyCue(0);
setScene(0);
renderClarification();

$('previousScene').onclick = () => setScene(sceneIndex - 1);
$('nextScene').onclick = () => setScene(sceneIndex + 1);
document.querySelector('.wordmark').onclick = event => { event.preventDefault(); setScene(0); };
$('closeSetup').onclick = () => toggleSetup(false);
$('playSource').onclick = () => playLocalFile(sourceFile);
$('replaySource').onclick = replayIntoModel;
$('useMic').onclick = startMicrophone;
$('playVoice').onclick = speakConfirmed;
$('sourceFile').onchange = event => {
  sourceFile = event.target.files[0] || null;
  $('sourceFileName').textContent = sourceFile?.name || 'Choose local WAV';
};
$('voiceFile').onchange = event => {
  voiceFile = event.target.files[0] || null;
  $('voiceFileName').textContent = voiceFile?.name || 'Choose local WAV';
};
$('fullscreen').onclick = () => document.documentElement.requestFullscreen?.();
document.addEventListener('keydown', handleKeyboard);
