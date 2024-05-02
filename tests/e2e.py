from haystack_embedded.haystack_embedded import HaystackEmbedded
import json
import time


db = HaystackEmbedded("test")

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
