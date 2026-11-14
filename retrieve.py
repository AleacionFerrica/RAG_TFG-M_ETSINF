import re
import sys
import unicodedata
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from index import EMBED_MODEL, VECTORS, embed_text, load_chunks

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 20  
RRF_K = 60  # constante estandar 

def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)

@cache

def resources():
    chunks = load_chunks()
    bm25 = BM25Okapi([tokenize(embed_text(c)) for c in chunks])
    return chunks, bm25, np.load(VECTORS), SentenceTransformer(EMBED_MODEL), CrossEncoder(RERANK_MODEL)

def retrieve(query: str, k: int = 5):
    chunks, bm25, vectors, embedder, reranker = resources()

    # 1. Dos rankings independientes sobre todos los fragmentos
    lexical = np.argsort(-bm25.get_scores(tokenize(query)))
    q = embedder.encode(query, normalize_embeddings=True)
    semantic = np.argsort(-(vectors @ q))

    # 2. Fusión RRF: premia lo que sale arriba en cualquiera de los dos
    rrf: dict[int, float] = {}
    for ranking in (lexical, semantic):
        for rank, idx in enumerate(ranking[:CANDIDATES]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1 / (RRF_K + rank + 1)
    candidates = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:CANDIDATES]

    # 3. El cross-encoder lee pregunta y fragmento juntos y puntúa de nuevo
    scores = reranker.predict([(query, chunks[i]["text"]) for i in candidates])
    best = sorted(zip(candidates, scores), key=lambda x: -x[1])[:k]
    return [{**chunks[i], "score": float(s)} for i, s in best]


if __name__ == "__main__":
    for hit in retrieve(" ".join(sys.argv[1:]) or "¿Cuándo es la defensa de diciembre?"):
        print(f"{hit['score']:.3f}  {hit['source']} p.{hit['page']}\n   {hit['text'][:150]!r}\n")