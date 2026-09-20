(function exposeVoiceBridgeDiff(global) {
  function words(text) {
    return String(text || '').trim().match(/\S+/g) || [];
  }

  function comparable(word) {
    return word.toLocaleLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
  }

  function alignWords(beforeText, afterText) {
    const before = words(beforeText);
    const after = words(afterText);
    const rows = before.length + 1;
    const columns = after.length + 1;
    const cost = Array.from({ length: rows }, () => Array(columns).fill(0));

    for (let row = 0; row < rows; row += 1) cost[row][0] = row;
    for (let column = 0; column < columns; column += 1) cost[0][column] = column;

    for (let row = 1; row < rows; row += 1) {
      for (let column = 1; column < columns; column += 1) {
        const equal = comparable(before[row - 1]) === comparable(after[column - 1]);
        cost[row][column] = Math.min(
          cost[row - 1][column - 1] + (equal ? 0 : 1),
          cost[row - 1][column] + 1,
          cost[row][column - 1] + 1,
        );
      }
    }

    const aligned = [];
    let row = before.length;
    let column = after.length;
    while (row > 0 || column > 0) {
      if (row > 0 && column > 0) {
        const equal = comparable(before[row - 1]) === comparable(after[column - 1]);
        const diagonalCost = cost[row - 1][column - 1] + (equal ? 0 : 1);
        if (cost[row][column] === diagonalCost) {
          aligned.push({
            before: before[row - 1],
            after: after[column - 1],
            kind: equal ? 'same' : 'replace',
          });
          row -= 1;
          column -= 1;
          continue;
        }
      }
      if (row > 0 && cost[row][column] === cost[row - 1][column] + 1) {
        aligned.push({ before: before[row - 1], after: null, kind: 'delete' });
        row -= 1;
      } else {
        aligned.push({ before: null, after: after[column - 1], kind: 'insert' });
        column -= 1;
      }
    }
    return aligned.reverse();
  }

  function summarize(alignment) {
    return alignment.reduce((summary, item) => {
      summary[item.kind] += 1;
      return summary;
    }, { same: 0, replace: 0, delete: 0, insert: 0 });
  }

  function wordNode(text, kind, index, emptyLabel) {
    const word = document.createElement('span');
    word.className = `diff-word ${kind}`;
    word.style.setProperty('--delay', `${index * 55}ms`);
    word.textContent = text || emptyLabel;
    return word;
  }

  function renderWordDiff(beforeTarget, afterTarget, alignment) {
    const beforeNodes = [];
    const afterNodes = [];
    alignment.forEach((item, index) => {
      beforeNodes.push(wordNode(item.before, item.kind, index, '∅'));
      afterNodes.push(wordNode(item.after, item.kind, index, '∅'));
    });
    beforeTarget.replaceChildren(...beforeNodes);
    afterTarget.replaceChildren(...afterNodes);
  }

  function runSelfChecks() {
    const replacement = alignWords('who were killed', 'wicked');
    const replacementKinds = replacement.map(item => item.kind).sort().join(',');
    if (replacementKinds !== 'delete,delete,replace') throw new Error('word diff replacement invariant failed');

    const insertion = alignWords('do give insulin', 'do not give insulin');
    const insertionSummary = summarize(insertion);
    if (insertionSummary.same !== 3 || insertionSummary.insert !== 1) {
      throw new Error('word diff insertion invariant failed');
    }
  }

  const api = { alignWords, summarize, renderWordDiff, runSelfChecks };
  global.VoiceBridgeDiff = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
