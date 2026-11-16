# adal-v1-active

The research harness for adaL v1. It starts as an exact copy of `timmetim/openGPT`
(2026-11-16): a word-level GPT built from scratch in plain numpy, with hand-written
backpropagation and Adam, and a local app to change it, run it and watch it train.
From here the two diverge: this copy is where the adaL research happens.

`glyphh-ai/adal-v0-dead` is the earlier adaL (a memory beside a frozen model). It is
dead and nothing from it is in here.

## Run it

```
uv sync
uv run python -m adal             # then open http://127.0.0.1:8643
uv run python -m adal --lan       # the same, for every device on your network
```

It works straight away on `tinytales`, a small made-up story corpus generated on
first use. For TinyStories: `uv run python -m adal fetch` (about 19 MB).

## The tabs

| Tab | What it is for |
| --- | --- |
| Learn | How the model works in 34 steps: math, Python and a worked example, built from the code as it stands. |
| Data | What the model reads: corpus numbers, the rare-word cut, the 90/10 hold-out, a tokenizer to try. |
| Train | Settings, Start / Pause / Stop, and a live dashboard: loss, held-out accuracy, gradient norm, learning rate, samples, and the checks that show it learned. |
| Generate | What a trained run writes, how sure it was of each word, and what every attention head looked at. |
| Runs | Every run is saved; lay their held-out loss on one chart. |
| Code | The source as it is on disk, and a button to re-check the gradients. |

## Doing research with it

The whole model is `adal/model.py`, one function per step. Edit it: the next run
reloads the file and the Learn tab rebuilds from it. After touching the maths, check
the gradients. Then run the same settings before and after and compare them in Runs.

```
uv run python -m adal check                       # hand-written gradients vs measured slopes
uv run python -m adal train steps=2000 n_layer=2  # train in the terminal, any setting as key=value
uv run pytest                                     # maths, training loop, every app call
uv run python tests/ui_check.py                   # draw every tab in headless Chrome, no server
```

## Layout

```
adal/model.py        the GPT: forward, backward, Adam, one function per step
adal/data.py         tokenize, vocabulary, hold-out, batches
adal/trainer.py      runs you can start, pause, stop and watch; saved in runs/
adal/walkthrough.py  builds the Learn tab from the live code
adal/api.py          every call the app makes, as plain functions
adal/server.py       a small local web server and the live stream
adal/static/         the page (plain JavaScript, Chart.js)
```
