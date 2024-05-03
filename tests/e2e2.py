from haystack_embedded.haystack_embedded import HaystackEmbedded
import json
import time
import requests
from tqdm.auto import tqdm


def embed_texts(texts):
    res = requests.post("REDACTED", json={"inputs": texts})
    res.raise_for_status()

    return res.json()


with open("test_texts.txt") as f:
    blob = f.read()


def make_chunks(blob):
    """

    Splits every 200 words into a chunk

    """

    words = blob.split()

    chunks = []

    for i in range(0, len(words), 96):
        chunks.append(" ".join(words[i : i + 96]))

    return chunks


CHUNKS = make_chunks(blob)

print(f"Number of chunks: {len(CHUNKS)}")

GROUP_SIZE = 8

start = time.perf_counter()

embeddings = []

for i in tqdm(range(0, len(CHUNKS), GROUP_SIZE)):
    embeddings.extend(embed_texts(CHUNKS[i : i + GROUP_SIZE]))

elapsed = (time.perf_counter() - start) * 1000

print(f"Embedding takes: {elapsed}ms")

print(f"Number of embeddings: {len(embeddings)}")


db = HaystackEmbedded("test")


for chunk, emb in tqdm(zip(CHUNKS, embeddings)):
    db.add_vector(
        emb,
        json.dumps(
            [{"key": "default", "value": "all"}, {"key": "text", "value": chunk}]
        ),
    )


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

for result in results:
    for item in result:
        print(item["value"])
        print()


b = db.save_state()

with open("state.bin", "wb") as f:
    f.write(bytearray(b))
