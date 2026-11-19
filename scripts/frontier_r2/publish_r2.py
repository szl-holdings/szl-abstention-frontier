"""Push one r2 run folder to its existing op1 repo in a single commit.

Uploads <out>/final/* and train_log.txt under r2/, plus any eval results under
eval/heldout-v1/results-r2 and eval/heldout-v2/results-r2, and writes
eval/frontier-r2/DONE.json. Never touches README.md, visibility, or gating.
"""
import argparse, json, os
from huggingface_hub import HfApi, CommitOperationAdd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)   # SZLHOLDINGS/salad-chaski-frontier-4b-op1
ap.add_argument("--out", required=True)
ap.add_argument("--status", required=True)  # DONE | BLOCKED
ap.add_argument("--reason", default="")
a = ap.parse_args()
ops = []
for root, _, files in os.walk(a.out):
    for f in files:
        p = os.path.join(root, f); rel = os.path.relpath(p, a.out)
        if rel.startswith("ckpt") or "checkpoint-" in rel:
            continue
        if rel.startswith("eval" + os.sep):
            dest = rel.replace(os.sep, "/")
        else:
            dest = "r2/" + rel.replace(os.sep, "/")
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=p))
done = {"status": a.status, "reason": a.reason, "run": "frontier-r2", "promotion": "HOLD",
        "signature": "UNSIGNED", "receipt_status": "MEASURED" if a.status == "DONE" else "NOT_MEASURED"}
ops.append(CommitOperationAdd(path_in_repo="eval/frontier-r2/DONE.json",
                              path_or_fileobj=json.dumps(done, indent=1).encode()))
c = HfApi().create_commit(a.repo, operations=ops, commit_message=f"frontier r2 ({a.status})")
print("COMMIT", a.repo, c.oid)
