(function exposeVoiceBridgeClarify(global) {
  function tokens(text) {
    return String(text || '').toLocaleLowerCase().match(/[\p{L}\p{N}']+/gu) || [];
  }

  function termCount(textTokens, term) {
    const termTokens = tokens(term);
    if (!termTokens.length) return 0;
    let count = 0;
    for (let index = 0; index <= textTokens.length - termTokens.length; index += 1) {
      if (termTokens.every((token, offset) => textTokens[index + offset] === token)) count += 1;
    }
    return count;
  }

  function criticalDisagreements(candidates, criticalTerms) {
    const candidateTokens = candidates.map(candidate => tokens(candidate.text ?? candidate));
    return criticalTerms.filter(term => {
      const counts = candidateTokens.map(candidate => termCount(candidate, term));
      return new Set(counts).size > 1;
    });
  }

  function mount(target, options) {
    const candidates = (options.candidates || []).slice(0, 3).map(candidate => (
      typeof candidate === 'string' ? { text: candidate } : candidate
    ));
    if (candidates.length < 2) throw new Error('Clarification needs at least two candidates');

    const disagreements = criticalDisagreements(candidates, options.criticalTerms || []);
    const shell = document.createElement('section');
    shell.className = 'clarify-shell';
    shell.innerHTML = `
      <div class="clarify-signal" aria-hidden="true"><span>?</span></div>
      <div class="clarify-content">
        <div class="clarify-kicker">VOICEBRIDGE STOPPED BEFORE GUESSING</div>
        <h2>Which did you mean?</h2>
        <p class="clarify-reason"></p>
        <div class="clarify-terms" aria-label="Critical disagreements"></div>
        <div class="clarify-options" role="group" aria-label="Transcript choices"></div>
        <p class="clarify-hint">PRESS A NUMBER TO CONFIRM THE EXACT WORDING</p>
      </div>`;

    const reason = shell.querySelector('.clarify-reason');
    reason.textContent = disagreements.length
      ? 'The candidates disagree on a safety-critical term.'
      : 'The evidence is weak enough that a fluent guess would be dishonest.';

    const termContainer = shell.querySelector('.clarify-terms');
    if (disagreements.length) {
      disagreements.forEach(term => {
        const chip = document.createElement('span');
        chip.textContent = term.toLocaleUpperCase();
        termContainer.append(chip);
      });
    } else {
      const chip = document.createElement('span');
      chip.textContent = `${Math.round((options.confidence ?? 0) * 100)}% CONFIDENCE`;
      termContainer.append(chip);
    }

    const buttons = candidates.map((candidate, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.key = String(index + 1);
      button.innerHTML = `<span>${index + 1}</span><strong></strong>`;
      button.querySelector('strong').textContent = candidate.text;
      button.onclick = () => select(index);
      shell.querySelector('.clarify-options').append(button);
      return button;
    });

    function select(index) {
      const selected = candidates[index];
      if (!selected) return;
      buttons.forEach((button, buttonIndex) => button.classList.toggle('selected', buttonIndex === index));
      options.onConfirm?.(selected.text, selected);
    }

    function handleKey(event) {
      if (event.repeat || !/^[1-3]$/.test(event.key)) return;
      const index = Number(event.key) - 1;
      if (candidates[index]) {
        event.preventDefault();
        select(index);
      }
    }

    target.replaceChildren(shell);
    document.addEventListener('keydown', handleKey);
    return {
      select,
      disagreements: [...disagreements],
      destroy() {
        document.removeEventListener('keydown', handleKey);
        shell.remove();
      },
    };
  }

  const api = { tokens, criticalDisagreements, mount };
  global.VoiceBridgeClarify = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
