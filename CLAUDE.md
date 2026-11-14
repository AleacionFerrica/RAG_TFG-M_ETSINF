# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

RAG over the TFG/TFM regulations of ETSINF (Valencia) and EPSA (Alcoy), UPV. Python, everything local: `BAAI/bge-m3` embeddings, BM25 + dense hybrid with RRF, `BAAI/bge-reranker-v2-m3` reranker, LLM via Ollama (`MODEL` in `ask.py`). No cloud LLM APIs.

## Commands (use the venv: `.RAGenv\Scripts\python`)
- `python ingest.py` → `data/chunks.jsonl` (PDF → chunks + metadata `META`: escuela, campus, curso)
- `python index.py` → `data/embeddings.npy`. Re-run after every ingest; the chunk count must match.
- `python ask.py` interactive chat (`/campus alcoy|valencia|todos`, `/salir`); `python ask.py <pregunta>` answers once
- `python evaluate.py` retrieval (Hit@5, MRR); `--respuestas` adds LLM answer metrics; `--ablacion` compares the `MODES` in `retrieve.py`

## Architecture
- `retrieve.py`: `retrieve(query, k, campus, mode)`. `campus` filters to that campus + "Ambos"; `detect_campus()` infers it from the question.
- `ask.py`: `ask()` auto-detects campus and returns `NO_ANSWER` without calling the LLM when the top rerank score < `MIN_SCORE`.
- `eval/preguntas.csv`: `respuesta_esperada` = key facts separated by `;`, `a|b` for variants; empty source/answer = unanswerable question.

## Corpus

`documents/` holds the source knowledge base: official ETSINF PDFs, all in Spanish. They cover:
- Academic calendar and key dates for 2026-27 (`Calendario-de-gestion-...`, `Fechas-Importantes-...`)
- Registering a TFG/TFM (`Guia-dar-Alta-...`, from 2019)
- TFG/TFM done during mobility/exchange programs
- Integrated TFG procedure for double degrees
- FAQ for 2026-27
- Defense tribunal procedure (2020) and defenses by videoconference over Microsoft Teams

Notes for ingestion/retrieval work:
- The documents come from different years (2019, 2020, 2026-27). Date-sensitive answers should prefer the 2026-27 documents, and keeping the source file and year as chunk metadata is useful for citations and for settling conflicts between documents.
- Retrieval, prompts and evaluation should handle Spanish-language content and queries.
- These are binary files. `.gitattributes` uses `* text=auto`, which detects PDFs as binary, so they are not altered.
