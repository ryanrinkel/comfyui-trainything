# TrainyThing for ComfyUI

Wire it after VAE Decode, press Queue, and the batch is on your phone as a
TrainyThing review queue: one card per image, swipe to keep or reject, say why.
The keepers, the notes, the seeds and the prompts come back into the graph.

It needs the TrainyThing daemon, which needs Node 22.5 or newer
(`npx -y trainything dashboard` once).

**Status: K0.** This is the client and the contract only (PLAN-COMFY K0); the
nodes arrive in K1. Nothing here shows up on the canvas yet.

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
| `nodes/` | the nodes (K1) |
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
```

MIT licensed.
