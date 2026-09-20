const tracks = {
  source: new Audio('fixtures/m02-source.wav'),
  personal: new Audio('fixtures/m02-personal-voice.wav'),
  generic: new Audio('fixtures/generic-tts.wav'),
};

const $ = id => document.getElementById(id);
let activeButton = null;
let toastTimer = null;

function setStatus(message) {
  $('sessionStatus').textContent = message;
}

function showToast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.add('show');
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 1800);
}

function stopTracks() {
  Object.values(tracks).forEach(track => {
    track.pause();
    track.currentTime = 0;
  });
  activeButton?.classList.remove('playing');
  activeButton = null;
}

async function playTrack(name, button, label) {
  stopTracks();
  const track = tracks[name];
  activeButton = button;
  button.classList.add('playing');
  setStatus(`Playing ${label}`);
  track.onended = () => {
    button.classList.remove('playing');
    activeButton = null;
    setStatus('Recorded comparison ready');
  };
  try {
    await track.play();
  } catch {
    button.classList.remove('playing');
    activeButton = null;
    setStatus('Audio unavailable');
    showToast('Audio could not be played');
  }
}

async function runComparison() {
  const button = $('runComparison');
  button.disabled = true;
  $('baselinePane').classList.add('processing');
  $('tunedPane').classList.add('processing');
  $('baselineState').textContent = 'Processing';
  $('tunedState').textContent = 'Queued';
  $('baselineTranscript').textContent = 'Listening…';
  $('tunedTranscript').textContent = 'Listening…';
  setStatus('Running recorded comparison');

  await new Promise(resolve => setTimeout(resolve, 420));
  $('baselinePane').classList.remove('processing');
  $('baselineState').textContent = 'Complete';
  $('baselineTranscript').textContent = '“who were killed”';
  $('tunedState').textContent = 'Processing';

  await new Promise(resolve => setTimeout(resolve, 360));
  $('tunedPane').classList.remove('processing');
  $('tunedState').textContent = 'Complete';
  $('tunedTranscript').textContent = '“wicked”';
  setStatus('Comparison complete');
  button.disabled = false;
  showToast('Comparison complete');
}

$('playSource').onclick = event => playTrack('source', event.currentTarget, 'source recording');
$('playPersonal').onclick = event => playTrack('personal', event.currentTarget, 'speaker voice');
$('playGeneric').onclick = event => playTrack('generic', event.currentTarget, 'generic TTS');
$('runComparison').onclick = runComparison;

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
