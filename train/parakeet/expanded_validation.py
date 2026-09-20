"""Expanded v1 development manifests; does not alter Phase 0 or active jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from contract import manifest
from contract.normalize import normalize, tokens
from train.parakeet.splits import phase1_split, write_jsonl

SPEAKERS = ('F01', 'M01', 'M03', 'M04', 'M05')
SEED = 'expanded-validation-v1'


def partition(rows, fraction=0.2):
    """Select whole utterance groups, stratified by speaker and prompt length.

    Keep all microphone views withheld; score one head-mic view per group.
    Array-only groups remain represented using their available microphone.
    This is utterance-disjoint, not globally prompt-disjoint validation.
    """
    if not 0 < fraction < 1:
        raise ValueError('fraction must lie between 0 and 1')
    groups = defaultdict(list)
    for row in rows:
        groups[row['utterance_group']].append(row)
    representatives = {}
    strata = defaultdict(list)
    for group, views in groups.items():
        if len({r['speaker_id'] for r in views}) != 1:
            raise ValueError('utterance group spans speakers')
        if len({normalize(r['text']) for r in views}) != 1:
            raise ValueError('microphone transcripts disagree')
        rep = min(views, key=lambda r: (r['mic'].lower() != 'headmic', r['utterance_id']))
        representatives[group] = rep
        if rep['speaker_id'] in SPEAKERS:
            kind = 'word' if len(tokens(rep['text'])) == 1 else 'sentence'
            strata[rep['speaker_id'], kind].append(group)
    selected = set()
    for key, keys in sorted(strata.items()):
        ordered = sorted(keys, key=lambda g: hashlib.sha256(f'{SEED}:{g}'.encode()).hexdigest())
        n = min(len(keys)-1, max(1, round(len(keys)*fraction))) if len(keys)>1 else 0
        selected.update(ordered[:n])
    train = [r for r in rows if r['utterance_group'] not in selected]
    heldout = [r for r in rows if r['utterance_group'] in selected]
    dev = [representatives[g] for g in sorted(selected)]
    return train, dev, heldout


def stage(output=None):
    # Read train/dev only; the sealed M02 manifest is neither loaded nor copied.
    expected = dict((name, digest) for digest, name in
                    (line.split() for line in Path('contract/MANIFEST_HASHES').read_text().splitlines()
                     if line.strip() and not line.startswith('#')))
    for name in ('torgo_dys_train.jsonl', 'torgo_dys_dev.jsonl'):
        if manifest.hash_manifest(name) != expected[name]:
            raise RuntimeError(f'frozen manifest mismatch: {name}')
    splits = phase1_split()
    train, dev, heldout = partition(splits.training)
    output = Path(output) if output else manifest.manifest_dir()
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        'expanded_v1_train.jsonl': train,
        'expanded_v1_dev.jsonl': dev + splits.validation,
        'expanded_v1_withheld_views.jsonl': heldout,
        'expanded_v1_smoke.jsonl': splits.baseline,
    }
    train_groups = {r['utterance_group'] for r in train}
    assert not train_groups & {r['utterance_group'] for r in dev+splits.validation+splits.baseline}
    assert not any(r['speaker_id']=='M02' for rows in datasets.values() for r in rows)
    receipt = {
        'version': SEED, 'fraction_per_speaker_and_kind': .2,
        'selection_metric': 'mean(F01 WER, M01 WER), computed separately by speaker',
        'regression_speakers': ['F04', 'M03', 'M04', 'M05'],
        'requires_fresh_training': True,
        'protocol': 'unseen utterances from familiar speakers; F04 is speaker-disjoint',
        'microphone_policy': 'headMic preferred; array-only groups use their available view',
        'datasets': {},
    }
    train_prompts = {normalize(r['text']) for r in train}
    for name, rows in datasets.items():
        path = output/name
        write_jsonl(path, sorted(rows, key=lambda r:r['utterance_id']))
        receipt['datasets'][name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'speakers': {speaker: {
                'clips': len(rs := [r for r in rows if r['speaker_id']==speaker]),
                'utterance_groups': len({r['utterance_group'] for r in rs}),
                'reference_words': sum(len(tokens(r['text'])) for r in rs),
                'word_clips': sum(len(tokens(r['text']))==1 for r in rs),
                'sentence_clips': sum(len(tokens(r['text']))>1 for r in rs),
                'non_headmic_clips': sum(r['mic'].lower()!='headmic' for r in rs),
                'clips_with_prompt_seen_in_training': sum(normalize(r['text']) in train_prompts for r in rs),
            } for speaker in sorted({r['speaker_id'] for r in rows})},
        }
    (output/'expanded_v1_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    print(json.dumps(stage(parser.parse_args().output), indent=2))
