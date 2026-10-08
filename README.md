# TrainyThing for ComfyUI

Wire it after VAE Decode, press Queue, and the batch is on your phone as a
TrainyThing review queue: one card per image, swipe to keep or reject, say why.
The keepers, the notes, the seeds and the prompts come back into the graph.

It needs the TrainyThing daemon, which needs Node 22.5 or newer
(`npx -y trainything dashboard` once).

**Status: K1.** Three nodes under *TrainyThing* on the canvas: **Review** (send
the batch, show the link and a QR, wait for the verdicts, pass on the keepers),
**Send** (the first half, for a run that should not wait) and **Wait** (the
second half). Load and Filter arrive in K2. The full README is K4.

![The Review node after a review](docs/screenshots/review-node.png)

Try it: copy `tests/fixtures/sprites/*.png` into ComfyUI's `input/` folder,
drag `examples/gate.json` onto the canvas, press Run, scan the QR.

## The nodes

| Node | Outputs |
|---|---|
| TrainyThing Review | `accepted`, `rejected`, `accepted_latent`, `images`, `labels`, `summary`, `notes`, `accepted_meta`, `accepted_indices` (a list), `link`, `slug` |
| TrainyThing Send | `queue` (only a Wait takes it), `link`, `slug` |
| TrainyThing Wait | as Review, minus `link` and `slug` |

An empty `accepted` or `rejected` stops only its own branch, silently.
`on_timeout: block` stops everything downstream. Interrupting a waiting node
leaves the queue open. Each finished wait writes
`output/trainything/<slug>/` (`results.json`, `blobs/`, `labels.jsonl`,
`summary.md`, and `preferences.jsonl` in pair mode).

## Where it finds the daemon

The same order as the TrainyThing CLI: `TRAINYTHING_URL` and
`TRAINYTHING_TOKEN` in ComfyUI's environment, then `base_url` / `port` and
`owner_token` in `$TRAINYTHING_HOME/config.json` (default
`~/.trainything/config.json`), then `http://127.0.0.1:4747`. The token is never
a widget, so it never lands in a PNG or a shared workflow.

## Layout

| Path | What |
|---|---|
| `__init__.py` | ComfyUI entrypoint (`comfy_entrypoint`, V3 API) |
| `nodes/` | the nodes: `review.py`, `send.py`, `wait.py`; `inputs.py` (the shared schema); `common.py` (the shared logic, importable without ComfyUI) |
| `examples/` | canvas workflows (drag onto ComfyUI) |
| `trainything_client.py` | the owner API client, standard library only, sync and async |
| `png.py` | IMAGE tensor <-> PNG with Save Image's `prompt` / `workflow` chunks |
| `walk.py` | the graph walk: seed, sampler, prompts, checkpoint, LoRAs |
| `split.py` | labels.jsonl -> accepted / rejected / contested / unjudged / per-option indices |
| `cards.py` | a batch -> cards; create-or-append by slug |
| `docs/V3-NOTES.md` | what was verified against ComfyUI, with file:line |
| `tests/` | pytest; see `tests/harness/README.md` |

## Tests

```
tests/harness/setup_comfy.sh   # once: CPU ComfyUI + venvs in ~/.cache/comfyui-trainything
tests/run.sh                   # everything, in both venvs
tests/harness/make_examples.py --shoot   # rebuild examples/*.json and the screenshots (headless Chromium)
```

MIT licensed.
