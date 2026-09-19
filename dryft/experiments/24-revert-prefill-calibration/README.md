# Restore the previous BF16 engine

At the user's request, revert the prefill backend calibration and 16,384-token
prefill graph limit introduced by fb78d96. The official run scored 939.8143
versus 946.4287 for e134e74. One official comparison is not proof of a universal
regression; the user requested the previous configuration.

The restored engine archive is byte-for-byte identical to the validated
e134e74 archive (SHA recorded in snapshot.json). Flash prefill and the
2,048-token graph cutoff are restored; tiled norm/RoPE, residual-norm fusion,
and projection calibration remain. Research artifacts are retained.

The separate INT8 MLP trial was stopped after its first two measured shapes
were 11.95% and 14.61% slower. Its 29 GPU tests and 21 kernel replay checks
passed, but full teacher-forced token verification had not completed. No INT8
production change or quantized Dryft submission was made.
