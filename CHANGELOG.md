# Changelog

## Unreleased

- The daemon is on npm: `trainything` 0.1.0 (2026-10-15), so `npx -y trainything
  dashboard` is the install line and the nodes' auto-start works as written.
  The README's pre-publish workaround is gone. The pack's own repository is
  public at github.com/ryanrinkel/comfyui-trainything.
- Every input and output has a plain-language tooltip (2026-10-15).
- The nodes come coloured (`web/trainything.js`, `WEB_DIRECTORY`), and a fresh
  Review or Send shows the essentials only: `images`, `title`, `prompt`,
  `mode`, `public`. The other inputs are flagged `advanced` in the schema and
  fold behind a **more settings** switch on the classic canvas; a loaded
  workflow that changed one of them opens unfolded. Load and Filter fold the
  same way. The inputs' order changed to put the essentials first, so
  workflows saved by 0.1.0 (positional `widgets_values`) load through the
  frontend's named values; the examples are regenerated.
- `tests/harness/groups.py` draws a titled group box around the TrainyThing
  nodes of a canvas workflow; the examples carry one.

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

**Review fixes** (an independent review of K0–K4; each has a test in
`tests/test_review_fixes*.py`, details in `docs/V3-NOTES.md`):

1. Wait and Review check `images_b` (pair) and the latent's size before the
   wait, not after it.
2. Appending to a queue made elsewhere: another mode is an error; the queue's
   options stand; a missing `min_reviews` is set from the widget by PATCH so
   the wait can end; a queue's own `min_reviews` is kept.
3. The wait retries a daemon that does not answer or answers 5xx (2, 4, 8 …
   30 s) until the timeout; 401 / 403 / 404 / 409 fail at once.
4. The split is computed before the run's folder is written; the folder is
   best effort and read by queue id (a renamed slug no longer breaks it).
5. Auto-start runs a `trainything` binary on the PATH first, `npx -y
   trainything list` only without one; the token goes along only with
   `TRAINYTHING_URL`.
6. `--disable-metadata` keeps the `prompt` / `workflow` chunks out of the
   uploaded PNGs.
7. `public` shares before any upload; a failed tunnel is the local link and
   a console line, never an error after the cards were added.
8. `accepted_meta` objects gain `position` (the position in the node's
   batch); Load leaves `index` as Send wrote it.
9. A local daemon whose `results?out=` file does not appear here (another
   filesystem) falls back to the bundle.
10. Load's fingerprint carries `last_activity_at`: an undo then a new verdict
    re-runs it.
11. The results bundle is streamed to a temp file, not read into memory.
12. Run ids carry a 4-hex random suffix.
