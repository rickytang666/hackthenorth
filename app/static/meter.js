(function exposeVoiceBridgeMeter(global) {
  function confidenceColor(confidence) {
    if (confidence >= 0.82) return '#355f82';
    if (confidence >= 0.62) return '#8a959d';
    return '#a34b42';
  }

  function mount(canvas, options = {}) {
    const context = canvas.getContext('2d');
    const sampleCount = options.sampleCount || 180;
    const samples = Array(sampleCount).fill(0);
    let tokens = [];
    let audioContext = null;
    let source = null;
    let analyser = null;
    let connectedStream = null;
    let animationFrame = null;
    let destroyed = false;

    function resize() {
      const ratio = Math.min(global.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function confidenceAt(progress) {
      if (!tokens.length) return 1;
      const tokenIndex = Math.min(Math.floor(progress * tokens.length), tokens.length - 1);
      return tokens[tokenIndex].confidence;
    }

    function draw() {
      if (destroyed) return;
      resize();
      const width = canvas.width;
      const height = canvas.height;
      const ratio = Math.min(global.devicePixelRatio || 1, 2);
      const transcriptHeight = 64 * ratio;
      const center = (height - transcriptHeight) * 0.5;
      context.clearRect(0, 0, width, height);

      context.strokeStyle = 'rgba(241,240,232,.08)';
      context.lineWidth = ratio;
      for (let line = 1; line < 5; line += 1) {
        const y = ((height - transcriptHeight) / 5) * line;
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      const step = width / (samples.length - 1);
      context.lineWidth = 2.4 * ratio;
      context.lineCap = 'round';
      for (let index = 1; index < samples.length; index += 1) {
        const progress = index / (samples.length - 1);
        const previousProgress = (index - 1) / (samples.length - 1);
        context.strokeStyle = confidenceColor(confidenceAt(progress));
        context.globalAlpha = .58 + confidenceAt(progress) * .42;
        context.beginPath();
        context.moveTo(previousProgress * width, center - samples[index - 1] * center * .82);
        context.lineTo(progress * width, center - samples[index] * center * .82);
        context.stroke();
      }
      context.globalAlpha = 1;

      const labelY = height - 21 * ratio;
      const gap = 9 * ratio;
      context.font = `800 ${11 * ratio}px "Avenir Next Condensed", "DIN Condensed", sans-serif`;
      context.textBaseline = 'middle';
      let cursor = 0;
      tokens.forEach(token => {
        const text = String(token.text).toLocaleUpperCase();
        const tokenWidth = context.measureText(text).width + 14 * ratio;
        if (cursor + tokenWidth > width) return;
        context.fillStyle = confidenceColor(token.confidence);
        context.globalAlpha = .14 + token.confidence * .18;
        context.fillRect(cursor, height - 40 * ratio, tokenWidth, 30 * ratio);
        context.globalAlpha = 1;
        context.fillStyle = confidenceColor(token.confidence);
        context.fillText(text, cursor + 7 * ratio, labelY);
        cursor += tokenWidth + gap;
      });

      animationFrame = global.requestAnimationFrame(draw);
    }

    function pushSample(level) {
      samples.shift();
      samples.push(Math.max(-1, Math.min(1, Number(level) || 0)));
    }

    function setSamples(nextSamples) {
      const sourceSamples = Array.from(nextSamples || []);
      samples.splice(0, samples.length, ...Array(sampleCount).fill(0));
      sourceSamples.slice(-sampleCount).forEach((sample, index, visible) => {
        samples[sampleCount - visible.length + index] = Math.max(-1, Math.min(1, Number(sample) || 0));
      });
    }

    function setTokens(nextTokens) {
      tokens = (nextTokens || []).map(token => ({
        text: token.text ?? token.token ?? '',
        confidence: Math.max(0, Math.min(1, Number(token.confidence ?? token.score ?? 0))),
      })).filter(token => token.text);
    }

    async function connect(stream) {
      disconnect();
      audioContext = new (global.AudioContext || global.webkitAudioContext)();
      connectedStream = stream;
      source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      const timeData = new Float32Array(analyser.fftSize);
      const sampleAudio = () => {
        if (!analyser || destroyed) return;
        analyser.getFloatTimeDomainData(timeData);
        let peak = 0;
        for (const value of timeData) peak = Math.max(peak, Math.abs(value));
        pushSample(Math.sin(performance.now() / 38) * peak);
        global.setTimeout(sampleAudio, 24);
      };
      sampleAudio();
    }

    function disconnect() {
      source?.disconnect();
      source = null;
      analyser = null;
      connectedStream?.getTracks().forEach(track => track.stop());
      connectedStream = null;
      if (audioContext) audioContext.close();
      audioContext = null;
    }

    function destroy() {
      destroyed = true;
      disconnect();
      if (animationFrame) global.cancelAnimationFrame(animationFrame);
    }

    draw();
    return { pushSample, setSamples, setTokens, connect, disconnect, destroy };
  }

  const api = { confidenceColor, mount };
  global.VoiceBridgeMeter = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
