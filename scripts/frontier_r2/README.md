# Frontier r2 (OOM-fixed relaunch of op1)

op1 (`SZLHOLDINGS/salad-chaski-frontier-4b-op1`, `-9b-op1`) ended
`BLOCKED TRAINING_FAILED rc=2`: CUDA OOM in `loss.backward` on 24 GB RTX 4090s.

r2 changes: micro-batch 1 + grad-accum 64, chunked assistant-only CE (no full
logits tensor), gradient checkpointing, an OOM guard that skips a micro-batch
(fail-closed after 20), and 4-bit QLoRA for 9B. Target: one 48 GB A6000, run
sequentially 4B then 9B (`run_queue_118.sh`), after the current job finishes.

Results are pushed in one commit per model under `r2/` and
`eval/frontier-r2/DONE.json` in the existing op1 repos. Evals must use the
heldout-v1 (46, probes sha `2f498461...`) and heldout-v2 (78) harness; if it is
not supplied the run is marked `BLOCKED EVAL_INPUTS_MISSING`, never scored by a
substitute rubric. Receipts: MEASURED / UNSIGNED; promotion HOLD. No card,
visibility, or gating changes.

`test`: `chunked_ce` matches full cross-entropy (checked on CPU).
