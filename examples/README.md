# Examples

Canvas workflows (PLAN-COMFY §5): drag one onto ComfyUI. They are built on
Load Image and the twelve sprites in `tests/fixtures/sprites`, so they run
with no model: copy those PNGs into ComfyUI's `input/` folder first. In a real
graph, VAE Decode's output goes where Batch Images is, and KSampler's latent
where Empty Latent is.

| File | What |
|---|---|
| `gate.json` | §5.1: Review inside the run; the keepers and the rejects saved, the notes, the summary and the keepers' meta shown |
| `two-runs-send.json` | §5.2 run 1: Send to `sprites-08` and finish; queue it again and the queue grows |
| `two-runs-load.json` | §5.2 run 2: TrainyThing Load reads `sprites-08` back; TrainyThing Filter `accepted` passes the keepers on (Save Image stands in for VAE Encode → KSampler); the summary and the keepers' meta shown |
| `ab-pair.json` | §5.3: Review in pair mode, A and B on one card; `accepted` is the winners |

`tests/harness/make_examples.py` rebuilds them through the frontend itself,
from `tests/harness/example_prompts.py`; `--check` then queues the two §5.2
workflows as the frontend would, with a scripted review between them. The dataset script
arrives in K4.
