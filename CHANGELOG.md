# Changelog

## 0.1.0 — 2026-10-08

The first version (PLAN-COMFY K0–K4). Not published to the Comfy Registry:
that waits for gate N, the final name.

**Five nodes**, under *TrainyThing* (ComfyUI V3 node API, `comfy_api.latest`,
ComfyUI ≥ 0.39.0):

- **TrainyThing Review**: a batch becomes a review queue (one card per image,
  labels, `meta` from the graph walk), the link and a QR on the node, an
  async wait with progress, cancel and a timeout, then the batch and its
  latent split by the verdicts: `accepted`, `rejected`, `accepted_latent`,
  `labels`, `summary`, `notes`, `accepted_meta`, `accepted_indices`.
- **TrainyThing Send**: the first half; creates or appends to a queue by slug
  and lets the run go on.
- **TrainyThing Wait**: the second half, wired to a Send's `queue`.
- **TrainyThing Load**: a later run reads a queue back by slug, images from
  the blob store, from a local or a remote daemon; re-runs only when the
  queue moved.
- **TrainyThing Filter**: any batch, its labels and a rule (`accepted`,
  `rejected`, `contested`, `unjudged`, `option`) → the subset and its latent.

Modes `single`, `pair` (A and B on one card, `preferences.jsonl`), `choose`
and `select` (options, counted in `labels` and `summary`). Every finished
wait and Load writes `output/trainything/<slug>/`. The token is read from the
environment or `~/.trainything/config.json`, never from a widget.

Also: `examples/` (five canvas workflows and `dataset.py`, keepers to
`<ref>.png` + `<ref>.txt`), a one-line warning on a ComfyUI older than the
pin, `pyproject.toml` with `[tool.comfy]`, and a registry publish workflow
that is not to be enabled before gate N.

**Gates passed** (the agent half; tests in `tests/`, run by `tests/run.sh`
on a CPU ComfyUI 0.39.0 against throwaway daemons):

- **K0**, the client and the contract: 12 images → 12 cards with refs, labels
  and `meta`; PNG `prompt` / `workflow` chunks round-trip; the graph walk on
  a saved SDXL + LoRA graph; URL and token precedence as the CLI's; the split
  rules in all four modes; the missing-daemon line.
- **K1**, Send, Wait and Review: the gate workflow, 8 kept and 4 rejected
  with notes, the run's folder; append across runs; cancel; timeout
  `partial` and `block`; zero rejects blocks only that branch; ComfyUI stays
  responsive during the wait.
- **K2**, Load and Filter: run 2 loads 24 images back within PNG rounding
  with the `meta` Send attached; Filter by every rule; Load re-runs only on a
  new judgment; a remote daemon through the bundle.
- **K3**, pair and options: twelve A/B pairs, `accepted` from both sides,
  `preferences.jsonl` of eleven lines, the skipped pair unjudged; a `choose`
  queue's counts and Filter per option.
- **K4**, packaging: `git clone` into a fresh ComfyUI's `custom_nodes/` lists
  the five nodes; the dataset script. The owner's real-sampler run is open
  (`docs/OWNER-RUN.md`).
