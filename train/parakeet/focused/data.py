"""Versioned prompt/group-disjoint development and held-out evaluation split."""
import hashlib
import json
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path

from contract import manifest
from contract.normalize import normalize, tokens
from train.parakeet.splits import phase1_split, write_jsonl
from train.parakeet.stage_training_data import _link_or_copy

SEED = 34019


def partition(rows):
    # Global text assignment prevents repeated prompts leaking across train/dev/test,
    # including other speakers and microphone views. F04 stays original speaker dev.
    dev_text, test_text = set(), set()
    protected = set()
    for speaker in ('F01', 'M01', 'M04'):
        for kind in ('word', 'sentence'):
            texts = {normalize(r['text']) for r in rows if r['speaker_id']==speaker and r['mic'].lower()=='headmic' and ('word' if len(tokens(r['text']))==1 else 'sentence')==kind}
            texts -= dev_text | test_text | protected
            ordered = sorted(texts, key=lambda t:hashlib.sha256(f'{SEED}:{speaker}:{t}'.encode()).hexdigest())
            n = max(2, round(len(ordered)*.15))
            dev_text.update(ordered[:n])
            if speaker != 'M04':
                test_text.update(ordered[n:2*n])
        protected.update(normalize(r['text']) for r in rows if r['speaker_id']==speaker)
    train = [r for r in rows if normalize(r['text']) not in dev_text|test_text and r['duration'] <= 30]
    dev = [r for r in rows if r['speaker_id'] in ('F01','M01','M04') and r['mic'].lower()=='headmic' and normalize(r['text']) in dev_text]
    test = [r for r in rows if r['speaker_id'] in ('F01','M01') and r['mic'].lower()=='headmic' and normalize(r['text']) in test_text]
    for a,b in [(train,dev),(train,test),(dev,test)]:
        assert not {r['utterance_group'] for r in a}&{r['utterance_group'] for r in b}
        assert not {normalize(r['text']) for r in a}&{normalize(r['text']) for r in b}
    return train,dev,test


class BalancedDraws:
    """Uniform speaker, then utterance group, then 70% preferred head microphone."""
    def __init__(self, rows, seed=SEED):
        self.rng=random.Random(seed)
        self.groups=defaultdict(lambda:defaultdict(list))
        for r in rows: self.groups[r['speaker_id']][r['utterance_group']].append(r)
        self.speakers=sorted(self.groups)
        self.keys={s:sorted(self.groups[s]) for s in self.speakers}
    def draw(self):
        s=self.rng.choice(self.speakers)
        rows=self.groups[s][self.rng.choice(self.keys[s])]
        head=[r for r in rows if r['mic'].lower()=='headmic']
        other=[r for r in rows if r['mic'].lower()!='headmic']
        return self.rng.choice(head if head and (not other or self.rng.random()<.7) else other or rows)


def stage():
    manifest.check_hashes('contract/MANIFEST_HASHES')
    splits=phase1_split()
    train,dev,test=partition(splits.training)
    root=Path('/private/tmp/voicebridge-focused/data/voicebridge');md=root/'manifests';md.mkdir(parents=True,exist_ok=True)
    purposes={'focused_train.jsonl':train,'focused_dev.jsonl':dev+splits.validation,'focused_test.jsonl':test,'focused_smoke.jsonl':splits.baseline}
    for name,rows in purposes.items():write_jsonl(md/name,rows)
    for name in ('torgo_dys_train.jsonl','torgo_dys_dev.jsonl'):shutil.copy2(manifest.manifest_dir()/name,md/name)
    for rows in purposes.values():
        for r in rows:
            dst=root/r['audio_filepath']
            if not dst.exists():_link_or_copy(manifest.data_root()/r['audio_filepath'],dst)
    receipt={}
    for name,rows in purposes.items():
        receipt[name]={'sha256':hashlib.sha256((md/name).read_bytes()).hexdigest(),'speakers':{s:{'rows':len(rs:=[r for r in rows if r['speaker_id']==s]),'words':sum(len(tokens(r['text'])) for r in rs),'groups':len({r['utterance_group'] for r in rs})} for s in sorted({r['speaker_id'] for r in rows})}}
    (root/'split_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':stage()
