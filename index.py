# Para cada chunk creado en la ingesta hacemos un embedding y guardamos 

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
CHUNKS = Path("data/chunks.jsonl")
VECTORS = Path("data/embeddings.npy")

def load_chunks():
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_text(chunk):
    # El nombre del documento da contexto a fragmentos que solos son ambiguos
    return f"{chunk['source']} (pág. {chunk['page']})\n{chunk['text']}"


def main():
    chunks = load_chunks() 
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([embed_text(chunk) for chunk in chunks], # El modelo transforma cada chunk en un embedding a modo de encoder
                            normalize_embeddings=True,
                            show_progress_bar=True)
    np.save(VECTORS, vectors)
    print(f"{vectors.shape[0]} vectores de dimensión {vectors.shape[1]} -> {VECTORS}")


if __name__ == "__main__":
    main()