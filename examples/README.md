# Examples

Canvas workflows: drag one onto ComfyUI. They are built on
Load Image and the twelve sprites in `tests/fixtures/sprites`, so they run
with no model: copy those PNGs into ComfyUI's `input/` folder first. In a real
graph, VAE Decode's output goes where Batch Images is, and KSampler's latent
where Empty Latent is.

| File | What |
|---|---|
| `gate.json` | Review inside the run; the keepers and the rejects saved, the notes, the summary and the keepers' meta shown |
| `two-runs-send.json` | Run 1: Send to `sprites-08` and finish; queue it again and the queue grows |
| `two-runs-load.json` | Run 2: TrainyThing Load reads `sprites-08` back; TrainyThing Filter `accepted` passes the keepers on (Save Image stands in for VAE Encode → KSampler); the summary and the keepers' meta shown |
| `ab-pair.json` | Review in pair mode, A and B on one card; `accepted` is the winners (from either side), `rejected` the losers; `preferences.jsonl` in `output/trainything/sprites-ab/` |
| `choose-reasons.json` | The reasons, counted: Review in choose mode with four reason tags (`hair:Hair colour, scarf:Scarf, hands:Hands, ok:Fine`); the summary counts them; TrainyThing Filter `option hair` → Save Image |

`tests/harness/make_examples.py` rebuilds them through the frontend itself,
from `tests/harness/example_prompts.py`; `--check` then queues the two workflows as the frontend would, with a scripted review between them, and
the two Review examples (`ab-pair`, `choose-reasons`) with a scripted review
while they wait.

## dataset.py

A dataset of keepers with captions, a script rather than a node: it
reads a run's folder (`results.json`, `labels.jsonl`, `blobs/`) and writes
`<ref>.png` + `<ref>.txt` pairs, the convention LoRA trainers read. The
caption is the card's `meta.prompt_text` (the graph walk's positive prompt, or
your `meta` input), else the card's label, else the queue's prompt.
Contested and unjudged cards are never taken. Standard library only.

```
python examples/dataset.py ComfyUI/output/trainything/sprites-08                      # keepers -> <folder>/dataset-accepted/
python examples/dataset.py ComfyUI/output/trainything/sprites-08 --out ~/lora/witch   # somewhere else
python examples/dataset.py ComfyUI/output/trainything/sprites-08 --rule rejected      # or option:<id> for choose / select
```

In `pair` mode it takes the winning side of each pair (`--rule rejected`: the
losing side); the card's `meta` describes A, so a B winner is captioned from
`meta.prompt_text_b` when your `meta` input set one, else from its label.

