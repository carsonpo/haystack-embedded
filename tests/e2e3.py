from haystack_embedded.haystack_embedded import HaystackEmbedded
import json
import time
import requests
from tqdm.auto import tqdm


with open("state.bin", "rb") as f:
    blob = f.read()


def embed_texts(texts):
    res = requests.post("REDACTED", json={"inputs": texts})
    res.raise_for_status()

    return res.json()


db = HaystackEmbedded("test")

db.load_state(blob)


query = "What did they use for activation functions"

query_embedding = embed_texts(
    ["Represent this sentence for searching relevant passages: " + query]
)[0]

start = time.perf_counter()

results = json.loads(
    db.query(
        query_embedding,
        json.dumps({"type": "Eq", "args": ["default", "all"]}),
        3,
    )
)

elapsed = (time.perf_counter() - start) * 1000

print(f"Query takes: {elapsed}ms")

print(results)

for result in results:
    for item in result:
        print(item["value"])
        print()
