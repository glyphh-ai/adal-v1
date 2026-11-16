# adal-v1-active

The research harness for adaL v1: a word-level GPT built from scratch in plain numpy
(hand-written backpropagation and Adam, any number of blocks and heads) and a local
app to run it, watch it train, and compare runs.

It began as a copy of `timmetim/openGPT` (2026-11-16) with the teaching tabs taken
out. `glyphh-ai/adal-v0-dead` is the earlier adaL (a memory beside a frozen model);
it is dead and nothing from it is in here.

## Run it

```
uv sync
uv run python -m adal fetch       # TinyStories validation split, about 19 MB, once
uv run python -m adal             # then open http://127.0.0.1:8643
uv run python -m adal --lan       # the same, for every device on your network
```

Press **Start run** with the settings as they are. Without TinyStories the app falls
back to `tinytales`, a small made-up corpus that needs no download.

## The tabs

| Tab | What it is for |
| --- | --- |
| Train | Settings, Start / Pause / Stop, and a live dashboard: loss, held-out accuracy, gradient norm, learning rate, text samples as it learns, the checks that show it learned, and a button to re-check the gradients. |
| Generate | What a trained run writes, how sure it was of each word, and what every attention head looked at. |
| Runs | Every run is saved in `runs/`; lay their held-out loss on one chart. |

## The default run

`corpus=tinystories min_freq=10 T=32 C=128 n_layer=2 n_head=4 B=32 lr=0.002
schedule=cosine warmup=200 clip=1 steps=8000`: 1.78M weights.

Measured 2026-11-16 on a Mac Studio CPU, seed 0: 6 minutes 47 seconds, about 20
steps a second. Held-out loss 2.886 (blind guessing is 8.6), held-out next-word
accuracy 40.0% (train 42.9%). What it writes from "once upon a time there was a":

> once upon a time there was a little bunny. he was very lonely and he was very
> adorable. every day, the bunny would hop and have fun. one day the seal saw a boy
> named jack. he was very scared of the giggling and laughter.

Sentences are grammatical and stay on a story for a line or two; it loses the thread
after that, as a 32-word context must. This is the number to beat.

## Doing research with it

The whole model is `adal/model.py`, one function per step. Edit it: the next run
reloads the file, nothing to restart. After touching the maths, check the gradients.
Then run the same settings before and after and compare them in Runs.

```
uv run python -m adal check                       # hand-written gradients vs measured slopes
uv run python -m adal train steps=2000 n_layer=1  # train in the terminal, any setting as key=value
uv run pytest                                     # maths, training loop, every app call
uv run python tests/ui_check.py                   # draw every tab in headless Chrome, no server
```

## Layout

```
adal/model.py      the GPT: forward, backward, Adam, one function per step
adal/data.py       tokenize, vocabulary, hold-out, batches
adal/trainer.py    runs you can start, pause, stop and watch; saved in runs/
adal/api.py        every call the app makes, as plain functions
adal/server.py     a small local web server and the live stream
adal/static/       the page (plain JavaScript, Chart.js)
```
