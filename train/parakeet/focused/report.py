"""Recompute saved prediction metrics and compare matched splits."""
import json
import sys
from pathlib import Path
import jiwer
from contract.normalize import normalize
from contract.evaluate import score


def main(root):
    data={}
    lines=['# Focused adaptation results','',
           'Both arms use fresh dim-32 adapters, balanced speaker/group sampling and greedy decoding. Checkpoint selection uses F01 development WER first, then M01, with other-speaker regression limits.','']
    for split in ['dev','test','smoke']:
        lines += [f'## {split}', '', '| Speaker | Clips / ref words | Frozen | Adapter-only | Adapter + last two blocks |', '|---|---:|---:|---:|---:|']
        data[split]={}
        for variant in ['baseline','adapter','partial']:
            folder='final_frozen' if variant=='baseline' and split=='test' else variant
            stem=split if variant=='baseline' else 'best_'+split
            rows=[json.loads(l) for l in (root/folder/(stem+'.jsonl')).read_text().splitlines()]
            expected=json.loads((root/folder/(stem+'_scores.json')).read_text())
            data[split][variant]={}
            for sp in sorted({r['speaker_id'] for r in rows}):
                rs=[r for r in rows if r['speaker_id']==sp];sc=score(rs)
                assert abs(sc['wer']-expected[sp]['wer'])<1e-12
                e=jiwer.process_words([normalize(r['text']) for r in rs],[normalize(r['prediction']) for r in rs])
                data[split][variant][sp]={**sc,'reference_words':e.hits+e.substitutions+e.deletions,'S':e.substitutions,'D':e.deletions,'I':e.insertions,'ids':sorted(r['utterance_id'] for r in rs)}
        for sp,base in data[split]['baseline'].items():
            for v in ['adapter','partial']:assert data[split][v][sp]['ids']==base['ids']
            values=' | '.join(f"{data[split][v][sp]['wer']:.2%}" for v in ['baseline','adapter','partial'])
            lines.append(f"| {sp} | {base['n']} / {base['reference_words']} | {values} |")
        lines.append('')
    lines += ['## Selected checkpoints','']
    for v in ['adapter','partial']:
        selected=json.loads((root/v/'selection.json').read_text());recipe=json.loads((root/v/'recipe.json').read_text())
        assert selected['frozen_weights_unchanged']
        lines.append(f"- {v}: step {selected['best_step']} of {selected['last_step']}; trainable counts {recipe['counts']}; frozen-weight integrity passed.")
    lines += ['', 'The separate test contains unseen prompts from known speakers, not unseen speakers or new recording sessions. Development results are used for selection. Original smoke samples are diagnostic only. No weights were deployed.']
    (root/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main(Path(sys.argv[1]))
