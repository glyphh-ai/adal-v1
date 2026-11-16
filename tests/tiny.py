"""Settings for a run small enough for a test: seconds on the made-up corpus.

Spelled out in full so the tests do not move when the app's defaults do.
"""

TINY = dict(corpus="tinytales", min_freq=1, T=12, C=32, mlp_mult=4, n_layer=1, n_head=2,
            B=32, lr=3e-3, schedule="constant", warmup=0, clip=0.0,
            eval_batches=4, sample_words=40, temperature=0.8, seed=0, dtype="float32")
