const PERSONAL_VOICE_URL = 'http://127.0.0.1:8770/v1/synthesize';
const paths = {
  baseline: {
    pane: document.getElementById('baselinePane'),
    runButton: document.getElementById('runBaseline'),
    state: document.getElementById('baselineState'),
    transcript: document.getElementById('baselineTranscript'),
    responseButton: document.getElementById('playBaselineReturn'),
    text: 'who were killed',
  },
  tuned: {
    pane: document.getElementById('tunedPane'),
    runButton: document.getElementById('runTuned'),
    state: document.getElementById('tunedState'),
    transcript: document.getElementById('tunedTranscript'),
    responseButton: document.getElementById('playTunedReturn'),
    text: 'wicked',
  },
};

const sessionStatus = document.getElementById('sessionStatus');
const sourceFileInput = document.getElementById('sourceFile');
const sourceTitle = document.getElementById('sourceTitle');
const sourceMeta = document.getElementById('sourceMeta');
const toast = document.getElementById('toast');
let sourceAudioUrl = null;
let activeSourceAudio = null;
let toastTimer = null;

function setStatus(message) {
  sessionStatus.textContent = message;
}

function showToast(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1800);
}

function stopCurrentAudio() {
  if (activeSourceAudio) {
    activeSourceAudio.pause();
    activeSourceAudio = null;
  }
  window.speechSynthesis?.cancel();
}

function setRunButtonsDisabled(disabled) {
  Object.values(paths).forEach(path => { path.runButton.disabled = disabled || !sourceAudioUrl; });
}

function playSourceRecording() {
  return new Promise((resolve, reject) => {
    if (!sourceAudioUrl) {
      reject(new Error('Select a source recording first'));
      return;
    }
    const audio = new Audio(sourceAudioUrl);
    activeSourceAudio = audio;
    audio.onended = () => {
      activeSourceAudio = null;
      resolve();
    };
    audio.onerror = () => {
      activeSourceAudio = null;
      reject(new Error('Source recording unavailable'));
    };
    audio.play().catch(reject);
  });
}

function speakWithGenericVoice(text) {
  return new Promise((resolve, reject) => {
    if (!window.speechSynthesis) {
      reject(new Error('Generic TTS unavailable'));
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.92;
    utterance.onend = resolve;
    utterance.onerror = () => reject(new Error('Generic TTS unavailable'));
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  });
}

async function speakWithPersonalVoice(text) {
  const response = await fetch(PERSONAL_VOICE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error(`Personal voice returned ${response.status}`);

  const context = new AudioContext();
  const buffer = await context.decodeAudioData(await response.arrayBuffer());
  await new Promise(resolve => {
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = resolve;
    source.start();
  });
  await context.close();
}

async function playReturnedSpeech(pathName) {
  const path = paths[pathName];
  path.responseButton.disabled = true;
  path.responseButton.classList.add('playing');
  setStatus(pathName === 'baseline' ? 'Playing generic returned speech' : 'Generating speaker-voice response');
  try {
    if (pathName === 'baseline') await speakWithGenericVoice(path.text);
    else await speakWithPersonalVoice(path.text);
    setStatus('Returned speech complete');
  } catch (error) {
    setStatus(error.message);
    showToast(error.message);
    throw error;
  } finally {
    path.responseButton.classList.remove('playing');
    path.responseButton.disabled = false;
  }
}

async function runPath(pathName) {
  const path = paths[pathName];
  stopCurrentAudio();
  setRunButtonsDisabled(true);
  path.pane.classList.remove('has-result');
  path.runButton.classList.add('running');
  path.responseButton.disabled = true;
  path.state.textContent = 'Playing input';
  path.transcript.textContent = 'Listening…';
  setStatus(`Running ${pathName === 'baseline' ? 'original model' : 'VoiceBridge'}`);

  try {
    await playSourceRecording();
    path.state.textContent = 'Transcribing';
    await new Promise(resolve => setTimeout(resolve, 260));
    path.transcript.textContent = `“${path.text}”`;
    path.pane.classList.add('has-result');
    path.responseButton.disabled = false;
    path.state.textContent = 'Returning speech';
    try {
      await playReturnedSpeech(pathName);
      path.state.textContent = 'Complete';
    } catch {
      path.state.textContent = 'Transcript ready';
    }
  } catch (error) {
    path.state.textContent = 'Input unavailable';
    path.transcript.textContent = 'Could not play the source recording';
    setStatus(error.message);
    showToast(error.message);
  } finally {
    path.runButton.classList.remove('running');
    setRunButtonsDisabled(false);
  }
}

paths.baseline.runButton.onclick = () => runPath('baseline');
paths.tuned.runButton.onclick = () => runPath('tuned');
paths.baseline.responseButton.onclick = () => playReturnedSpeech('baseline').catch(() => {});
paths.tuned.responseButton.onclick = () => playReturnedSpeech('tuned').catch(() => {});

sourceFileInput.onchange = () => {
  const file = sourceFileInput.files[0];
  if (!file) return;
  if (sourceAudioUrl) URL.revokeObjectURL(sourceAudioUrl);
  sourceAudioUrl = URL.createObjectURL(file);
  sourceTitle.textContent = file.name;
  sourceMeta.textContent = `${(file.size / 1024).toFixed(0)} KB · local file · shared by both paths`;
  Object.values(paths).forEach(path => {
    path.state.textContent = 'Ready';
    path.runButton.disabled = false;
  });
  setStatus('Local recording ready');
};

document.querySelectorAll('[data-choice]').forEach(button => {
  button.onclick = () => {
    document.querySelectorAll('[data-choice]').forEach(option => option.classList.toggle('selected', option === button));
    setStatus('Exact wording confirmed');
    showToast('Meaning confirmed');
  };
});

document.addEventListener('keydown', event => {
  if (event.key === '1' || event.key === '2') {
    document.querySelector(`[data-choice]:nth-child(${event.key})`)?.click();
  }
});
