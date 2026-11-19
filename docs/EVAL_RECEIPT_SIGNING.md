# Eval receipt signing (Sigstore keyless, GitHub artifact attestations)

Status: **proposal, draft.** Merging this file and `.github/workflows/eval-receipt-attest.yml`
needs Stephen Lutar's separate, explicit approval, because after merge the workflow starts
writing **public** entries to the Sigstore transparency log (Rekor) under this repository's
workflow identity. Nothing here changes a promotion gate, a HOLD label, or a
MEASURED/LIVE/`publication_eligible` field. Clearing HOLD on signed receipts is a separate,
later PR.

## What it does
`eval-receipt-attest.yml` produces an [in-toto Statement](https://github.com/in-toto/attestation)
in a [DSSE](https://github.com/secure-systems-lab/dsse) envelope, signed keylessly with
[Sigstore](https://docs.sigstore.dev/) through
[`actions/attest`](https://github.com/actions/attest) (pinned to v4.2.2 by commit SHA).

| Field | Value |
|---|---|
| Subject | the receipt file name and its sha256 |
| Predicate type | `https://szl-holdings.dev/eval-receipt/v1` |
| Predicate | the source (path, URL or push commit), each subject's sha256, the receipt's declared `schema`/`status`/`promotion`/`signature` fields copied as they are (never modified), and the run's repository, ref, commit, workflow_ref and run id |
| Signer identity | `szl-holdings/szl-abstention-frontier/.github/workflows/eval-receipt-attest.yml@refs/heads/main` (GitHub OIDC) |
| Permissions | `contents: read`, `id-token: write`, `attestations: write`; no secrets, no keys |

What an attestation proves: these exact bytes were seen by this repository's workflow on
`main` at a given commit and time. It does **not** prove that the eval scores are right,
that the model is fit to promote, or that the probe/eval/scoring files listed inside a receipt
match (those must be uploaded and attested too).

## Triggers
- `workflow_dispatch` with inputs `receipt` (a repo-relative path, or a public `https://` URL)
  and `sha256` (64 hex). If the bytes do not hash to `sha256`, the job fails and signs nothing.
  Private Hugging Face files are **not** fetched (no tokens are used). Download them through
  authorized access, commit them, or publish them, then attest them by path.
- `push` to `main` that adds or modifies `receipts/**`, `eval/**/receipt.json` or
  `eval/**/summary.json`. Each changed file is attested. If a sidecar `<file>.sha256` exists, it
  is enforced, and a mismatch fails the run.
- Any run on a ref other than `refs/heads/main` is skipped, so the identity stays pinned to main.
  The workflow has no `pull_request` trigger, so PRs never create attestations.

## Verify
With the GitHub CLI (`gh` 2.49 or later):

```bash
gh attestation verify receipts/abst_eval_r4r2_summary.json \
  --repo szl-holdings/szl-abstention-frontier \
  --signer-workflow szl-holdings/szl-abstention-frontier/.github/workflows/eval-receipt-attest.yml \
  --source-ref refs/heads/main \
  --predicate-type https://szl-holdings.dev/eval-receipt/v1
```

Add `--format json` to read the predicate. For a file attested from a URL, run the same
command on the downloaded bytes; the subject is matched by digest.

Offline / without `gh`: download the bundle with
`gh attestation download <file> --repo szl-holdings/szl-abstention-frontier` and verify it with
`cosign verify-blob-attestation --bundle <bundle.jsonl> --new-bundle-format
--certificate-oidc-issuer https://token.actions.githubusercontent.com
--certificate-identity https://github.com/szl-holdings/szl-abstention-frontier/.github/workflows/eval-receipt-attest.yml@refs/heads/main
--type https://szl-holdings.dev/eval-receipt/v1 <file>`.

A string `"signature": "SIGNED"` inside a receipt is **not** a signature. Only a verified
attestation as above counts.

## Rollback
Revert the PR, or disable the workflow in the Actions tab. Entries already written to Rekor
are append-only and public; they cannot be deleted, only stop being added.

## Not in scope
- No promotion-gate change; HOLD stays HOLD.
- No signing of the probe set, eval script or scoring script until they are uploaded.
- No private-asset upload, no secrets, no long-lived keys.
