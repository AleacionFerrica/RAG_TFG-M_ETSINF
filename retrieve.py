import re
import sys
import unicodedata
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from index import EMBED_MODEL, VECTORS, embed_text, load_chunks

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 20  # cuántos pasan al reranker
RRF_K = 60  # constante estándar de Reciprocal Rank Fusion
