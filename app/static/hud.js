(function exposeVoiceBridgeLatencyHud(global) {
  const STAGES = [
    { id: 'capture', label: 'Capture' },
    { id: 'asr', label: 'ASR' },
    { id: 'verify', label: 'Verify' },
    { id: 'synth', label: 'Synth' },
  ];

  function mount(target, options = {}) {
    const budgetMs = options.budgetMs || 1500;
    const provider = options.provider || 'Baseten';
    const clock = options.clock || (() => performance.now());
    const durations = Object.fromEntries(STAGES.map(stage => [stage.id, null]));
    const started = Object.fromEntries(STAGES.map(stage => [stage.id, null]));
    let currentStage = null;
    let completed = false;

    const shell = document.createElement('aside');
    shell.className = 'latency-hud';
    shell.innerHTML = `
      <div class="latency-hud__topline">
        <span class="latency-hud__provider"></span>
        <span class="latency-hud__state">READY</span>
      </div>
      <div class="latency-hud__headline">
        <strong class="latency-hud__total">0</strong>
        <span>MS TO FIRST AUDIO</span>
        <b class="latency-hud__budget"></b>
      </div>
      <div class="latency-hud__rail" aria-label="Latency budget usage">
        <div class="latency-hud__segments"></div>
        <i class="latency-hud__limit" aria-hidden="true"></i>
      </div>
      <div class="latency-hud__stages"></div>`;
    shell.querySelector('.latency-hud__provider').textContent = `${provider.toLocaleUpperCase()} LIVE PATH`;
    shell.querySelector('.latency-hud__budget').textContent = `${(budgetMs / 1000).toFixed(1)}S BUDGET`;
    target.replaceChildren(shell);

    const segmentRoot = shell.querySelector('.latency-hud__segments');
    const stageRoot = shell.querySelector('.latency-hud__stages');
    STAGES.forEach((stage, index) => {
      const segment = document.createElement('span');
      segment.dataset.stage = stage.id;
      segment.style.setProperty('--stage-index', index);
      segmentRoot.append(segment);

      const row = document.createElement('div');
      row.className = 'latency-hud__stage';
      row.dataset.stage = stage.id;
      row.innerHTML = `<span>${String(index + 1).padStart(2, '0')} · ${stage.label}</span><strong>WAIT</strong>`;
      stageRoot.append(row);
    });

    function total() {
      return STAGES.reduce((sum, stage) => sum + (durations[stage.id] || 0), 0);
    }

    function render() {
      const elapsed = total();
      const budgetPercent = Math.min((elapsed / budgetMs) * 100, 100);
      shell.style.setProperty('--budget-used', `${budgetPercent}%`);
      shell.style.setProperty('--budget-ratio', Math.min(elapsed / budgetMs, 1));
      shell.classList.toggle('is-over', elapsed > budgetMs);
      shell.classList.toggle('is-complete', completed);
      shell.querySelector('.latency-hud__total').textContent = Math.round(elapsed).toLocaleString();
      shell.querySelector('.latency-hud__state').textContent = completed
        ? (elapsed <= budgetMs ? `UNDER BY ${Math.round(budgetMs - elapsed)} MS` : `OVER BY ${Math.round(elapsed - budgetMs)} MS`)
        : (currentStage ? `${currentStage.toLocaleUpperCase()} RUNNING` : 'READY');

      let offset = 0;
      STAGES.forEach(stage => {
        const duration = durations[stage.id];
        const segment = segmentRoot.querySelector(`[data-stage="${stage.id}"]`);
        const row = stageRoot.querySelector(`[data-stage="${stage.id}"]`);
        const width = duration === null ? 0 : Math.max((duration / budgetMs) * 100, 0.7);
        segment.style.left = `${Math.min(offset, 100)}%`;
        segment.style.width = `${Math.min(width, Math.max(100 - offset, 0))}%`;
        segment.classList.toggle('active', currentStage === stage.id);
        row.classList.toggle('active', currentStage === stage.id);
        row.classList.toggle('done', duration !== null);
        row.querySelector('strong').textContent = duration === null ? (currentStage === stage.id ? 'LIVE' : 'WAIT') : `${Math.round(duration)} MS`;
        offset += width;
      });
    }

    function reset() {
      STAGES.forEach(stage => {
        durations[stage.id] = null;
        started[stage.id] = null;
      });
      currentStage = null;
      completed = false;
      render();
    }

    function begin(stageId) {
      if (!(stageId in durations)) throw new Error(`Unknown latency stage: ${stageId}`);
      currentStage = stageId;
      started[stageId] = clock();
      completed = false;
      render();
    }

    function record(stageId, durationMs) {
      if (!(stageId in durations)) throw new Error(`Unknown latency stage: ${stageId}`);
      durations[stageId] = Math.max(0, Number(durationMs));
      if (!Number.isFinite(durations[stageId])) throw new Error(`Invalid latency duration: ${durationMs}`);
      if (currentStage === stageId) currentStage = null;
      render();
    }

    function end(stageId) {
      if (started[stageId] === null) throw new Error(`Latency stage was not started: ${stageId}`);
      record(stageId, clock() - started[stageId]);
      return durations[stageId];
    }

    function finish() {
      currentStage = null;
      completed = true;
      render();
      return snapshot();
    }

    function snapshot() {
      return { budgetMs, totalMs: total(), stages: { ...durations } };
    }

    reset();
    return { begin, end, record, finish, reset, snapshot };
  }

  const api = { STAGES, mount };
  global.VoiceBridgeLatencyHud = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
