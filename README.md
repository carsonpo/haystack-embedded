# Haystack Embedded

> np.array++

This is a companion library for [HaystackDB](https://github.com/carsonpo/haystackdb), meant to be as simple as np.array but with lots of other helpful functionality.

## Features

- In memory, local first vector DB
- Binary embeddings by default (soon int8 reranking)
- Metadata filtering for queries
- Python bindings out of the box (soon WASM as well)

## Benchmarks

> On a MacBook with an M2, 1024 dimension, binary quantized.

> FAISS is using a flat index, so brute force, just like Haystack.

TLDR is Haystack Embedded is ~2.5-3x faster than normal Haystack (on disk), and 25-35x faster than FAISS (in memory).

```
100,000 Vectors
Haystack Embedded - 1.04ms
Haystack          — 3.44ms
FAISS             — 29.67ms

500,000 Vectors
Haystack Embedded - 5.27ms
Haystack          — 11.98ms
FAISS             - 146.50ms

1,000,000 Vectors
Haystack Embedded - 8.12ms
Haystack          — 22.65ms
FAISS             — 293.91ms
```

## Roadmap

- **Quality benchmarks** (this is in progress)
- Int8 reranking
- WASM bindings
- Wrapper library so it can be pip installable
- Abstract the weird metadata filtering syntax
- Syncing with [HaystackDB](https://github.com/carsonpo/haystackdb) from the cloud (or your self hosted instances)
- Kmeans clustering of vectors for centroid based ANN search

## Quickstart

Compile the bindings with `maturin build --release` (need to have Rust installed with maturin), and pip install the resulting wheel in /target

For real world vectors, I'd recommend using [MixedBreadAI's model](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)

```python
from haystack_embedded.haystack_embedded import HaystackEmbedded
import json
import time


db = HaystackEmbedded("your_namespace_id")

NUM_VECTORS = 1_000_000

db.batch_add_vectors(
    [[-1.0 for _ in range(1024)] for _ in range(NUM_VECTORS)],
    json.dumps([[{"key": "default", "value": "all"}] for _ in range(NUM_VECTORS)]),
)

# db.add_vector(
#     [1.0 for _ in range(1024)],
#     json.dumps(
#         [{"key": "default", "value": "all"}, {"key": "other", "value": "value"}]
#     ),
# )

# print(
#     db.query(
#         [1.0 for _ in range(1024)],
#         json.dumps({"type": "Eq", "args": ["default", "all"]}),
#         1,
#     )
# )

# prints `{"key": "default", "value": "all"}, {"key": "other", "value": "value"}`


NUM_RUNS = 100

start = time.perf_counter()

for _ in range(NUM_RUNS):
    db.query(
        [1.0 for _ in range(1024)],
        json.dumps({"type": "Eq", "args": ["default", "all"]}),
        1,
    )

elapsed = (time.perf_counter() - start) * 1000 / NUM_RUNS

print(f"Query takes: {elapsed}ms")
```
