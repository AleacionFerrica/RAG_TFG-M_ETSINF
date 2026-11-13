# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is at its very start: there is no source code, build system, dependencies, or tests yet. Based on the repo name (`RAG_TFG-M_ETSINF`), the goal is a Retrieval-Augmented Generation (RAG) system over the regulations and procedures for the TFG/TFM (Trabajo Fin de Grado / Trabajo Fin de Máster, the bachelor's and master's final projects) at ETSINF (Escuela Técnica Superior de Ingeniería Informática, UPV). The language, framework, embedding model, vector store and LLM provider have not been chosen yet. Don't assume any of them; ask, or follow what the code establishes once it exists.

Update this file with build/test/run commands and architecture notes once code is added.

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
