# Parakeet 1.1B focused-adaptation handoff

## Decision

Proceed with `nvidia/parakeet-tdt-1.1b` using the dim-32 encoder adapters and
the final two FastConformer encoder blocks unfrozen. Prediction and joint
networks remain frozen, decoding remains greedy, and the pretrained encoder
weights use one tenth of the adapter learning rate.

The selected checkpoint is the `partial` arm from Baseten job `qz801kq`:

- selected optimizer step: 800
- training stopped on plateau at step 1050
- trainable adapter parameters: 2,838,528
- trainable pretrained parameters: 50,415,616
- adjusted encoder blocks: 40 and 41 of 42, zero-based
- adapter learning rate: 3e-5
- pretrained learning rate: 3e-6
- frozen-weight integrity check: passed

The full `.nemo` checkpoint is required. Exporting only the adapter would lose
the updates to encoder blocks 40 and 41.

## Results

All values below are WER with greedy decoding.

| Evaluation | Frozen 1.1B | Adapter only | Adapter + final two blocks |
|---|---:|---:|---:|
| F01 development | 61.36% | 25.00% | **22.73%** |
| F04 development | 16.48% | 8.32% | **7.85%** |
| M01 development | 63.85% | **43.85%** | **43.85%** |
| M04 development | 74.59% | **48.36%** | 49.18% |
| F01 held-out prompts | 77.78% | **36.11%** | 44.44% |
| M01 held-out prompts | 77.04% | **37.04%** | 41.48% |
| F01 four-clip smoke set | 47.83% | **26.09%** | 39.13% |

The product decision is to carry the final-two-block checkpoint forward. The
evaluation caveat must remain visible: it won F01 development and F04, but the
adapter-only checkpoint generalized better on the inspected F01/M01 held-out
prompts and the F01 smoke set. The held-out prompt set has now been inspected
and must not be described as a blind final test in later work.

## Implementation

The implementation is contained in `train/parakeet/`:

- `focused/config_1b.py`: Baseten H100 training job
- `focused/run.py`: adapter creation, selective unfreezing, balanced training,
  F01-first checkpoint selection, plateau stopping, evaluation, and export
- `focused/data.py`: deterministic disjoint F01/M01 splits and balanced sampler
- `focused/report.py`: row-level metric verification and comparison report
- `focused/test_data.py`: split, sampling, checkpoint-selection, and guardrail tests
- `decode.py`, `splits.py`, and `stage_training_data.py`: shared dependencies

Run the local protocol checks with:

```sh
python -m unittest train.parakeet.focused.test_data -v
```

Launch the 1.1B comparison with:

```sh
python -m train.parakeet.focused.data
truss train push train/parakeet/focused/config_1b.py --remote baseten-team34 --team 34
```

## Checkpoint and publication

Baseten job: <https://app.baseten.co/training/qel6803/logs/qz801kq>

The selected checkpoint artifact is `results/partial/best.nemo` in the job's
checkpoint storage. Its exact size is 4,294,574,080 bytes, just under 4 GiB.
The prepared local destination is:

```text
checkpoints/parakeet-tdt-1.1b-focused-partial/best.nemo
```

Verified local artifact:

```text
size:   4,294,574,080 bytes
sha256: 55e407ff5c146fe09ca1fb308548f5a726ef36ad23340435f3e8186b5c0041a9
```

`checkpoints/` and `*.nemo` are intentionally ignored by Git. Do not force-add
this checkpoint to the source repository. Ordinary GitHub Git objects are
limited to 100 MiB; GitHub Free and Pro also limit each Git LFS object to 2 GB.
Publish the checkpoint to a model registry such as Hugging Face after authenticating
an account, then record its immutable repository revision and SHA-256 here.

## Remaining publication steps

1. Authenticate the intended Hugging Face organization or another model registry.
2. Create a model repository with the training recipe, split limitations,
   evaluation table, base-model attribution, and license metadata.
3. Upload `best.nemo` and pin the returned revision in this handoff.
4. Load the uploaded checkpoint in a clean environment and reproduce at least
   the F01/M01 smoke predictions before deployment.
