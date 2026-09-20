# Fixed dispatch; no startup calibration

Removed both decode and speculative-verification calls to the timing-based calibration stage. CUDA/Triton initialization and production CUDA-graph capture remain. No competing kernels are timed at runtime. Existing measured projection configurations remain the defaults; down/next-layer normalization fusion is fixed on for 2–32 rows, with split-down for the final residual. Ordinary single-query attention uses SGLang splits 8 at batch 8, splits 4 at batch 16, and the fused local implementation otherwise. Multi-query verification retains position-aware attention.

The offline calibration helpers remain available for research but Engine does not call them. The validation run replaces calibration with a function that raises, and asserts that the dynamic-choice and forced-choice tables remain empty throughout all workloads.

H100 validation covered (batch, prompt, output) shapes (1,512,32), (4,2048,32), (16,512,128), (8,1024,64), and (32,256,32). All 3,744 candidate positions passed native own-prefix checks; worst logit deficit was 0.25. Fresh prompts were generated three times after warmup. Shape warmups took 2.08, 3.99, 2.97, 4.22, and 10.47 seconds respectively on an existing Triton cache. These are not clean-cache compilation measurements, and calibration was untimed by the official benchmark.

Local tests passed (37 tests; 28 skipped due to missing CUDA/dependencies). Python parsing, git diff whitespace checks, and Dryft engine packaging validation passed.

The user authorized uploading source and starting a public benchmark. The archive POST returned HTTP 405 / method_not_allowed, creating no submission or run. Live https://htn.dryft.ai/docs confirms that public runs have been retired and submissions now require a push to the connected repository default branch, which starts an official run. The user subsequently authorized pushing this version to main and starting the official run.
