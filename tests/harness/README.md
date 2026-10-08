# The test harness

Everything lives outside the repo, in `~/.cache/comfyui-trainything/`
(override with `COMFY_HARNESS`):

| Path | What |
|---|---|
| `venv/` | Python venv: torch CPU wheels (from `download.pytorch.org/whl/cpu`, no CUDA), ComfyUI's requirements, `segno`, `pytest` |
| `pure-venv/` | Pillow, numpy and pytest only: stands in for a plain system Python |
| `ComfyUI/` | a clone of `comfyanonymous/ComfyUI` |
| `ComfyUI/custom_nodes/comfyui-trainything` | symlink to this repo |

## Set up (idempotent)

```
tests/harness/setup_comfy.sh
```

## Run the tests

```
tests/run.sh                 # harness venv (all), then pure venv (ComfyUI checks skip)
~/.cache/comfyui-trainything/venv/bin/python -m pytest -q          # harness venv only
~/.cache/comfyui-trainything/venv/bin/python -m pytest -q -k walk  # a subset
```

The daemon tests start a throwaway TrainyThing daemon from the daemon's repo
(`TRAINYTHING_REPO`, default `../trainything`; it must have `dist/cli.js`,
i.e. `pnpm build` once): `TRAINYTHING_HOME` in a pytest temp dir, a random port
in 20000-40000, killed after the session. They never talk to port 4747 or read
`~/.trainything`.

## By hand

A throwaway daemon:

```
export TRAINYTHING_HOME="$(mktemp -d)/home"
PORT=$(( 20000 + RANDOM % 20001 ))
node ../trainything/dist/cli.js serve --daemon --port "$PORT"
export TRAINYTHING_URL="http://127.0.0.1:$PORT"
# ... use it; the token is in $TRAINYTHING_HOME/config.json
kill "$(cat "$TRAINYTHING_HOME/server.pid")"
```

ComfyUI on CPU, with the pack loaded:

```
cd ~/.cache/comfyui-trainything/ComfyUI
PORT=$(( 18000 + RANDOM % 2000 ))
../venv/bin/python main.py --cpu --port "$PORT" --listen 127.0.0.1 --disable-auto-launch
# another shell: curl -s http://127.0.0.1:$PORT/system_stats
# the log lists custom_nodes/comfyui-trainything under "Import times for custom nodes"
```

Run it with both `TRAINYTHING_URL` and `TRAINYTHING_HOME` exported to point
it at the throwaway daemon instead of the real one.
