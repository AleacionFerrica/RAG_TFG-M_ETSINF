# Guía del TFG/TFM de la ETSINF
Un asistente que responde dudas sobre el TFG y el TFM de la UPV a partir de la normativa oficial [ETSINF](https://www.upv.es/entidades/etsinf/trabajo-fin-de-grado/). Funciona en tu ordenador: no usa ninguna API externa y no envía las preguntas a ningún servidor.

## Por qué

Cuando llega el TFG, la información está repartida entre calendarios, FAQs, guías de Ebrón y procedimientos de distintos años. Saber hasta cuándo puedes pedir la defensa, qué documentos hay que subir o qué pasa si cambias de tema obliga a abrir varios PDF y a comprobar que son del curso correcto y de tu escuela.

Mediante un RAG le haces la pregunta en castellano y te responde con el dato y el documento y la página de donde lo ha sacado, para que puedas comprobarlo.

```
[todos] > ¿Hasta cuándo puedo solicitar la defensa en la convocatoria de marzo?
La fecha límite para solicitar la defensa en la convocatoria de marzo es el 22 de febrero de 2027 [1].

Fuentes:
  [1] Fechas-Importantes-TFG-2026-2027.pdf, pág. 1 (Valencia)
```

## Cómo funciona

Es un RAG (*Retrieval-Augmented Generation*) sencillo, en cinco scripts de Python:

1. **`ingest.py`**: convierte los PDF de `documents/` a texto con `pymupdf4llm` y los divide en fragmentos de unos 1000 caracteres, respetando los párrafos. Cada fragmento guarda el documento, la página, la escuela, el campus y el curso.
2. **`index.py`**: calcula un embedding de cada fragmento con `BAAI/bge-m3`.
3. **`retrieve.py`**: busca los fragmentos más relevantes en tres pasos:
   - una búsqueda léxica (BM25) y otra semántica (embeddings) por separado;
   - las dos listas se combinan con *Reciprocal Rank Fusion*;
   - un *reranker* (`BAAI/bge-reranker-v2-m3`) reordena los 20 mejores y se quedan los 5 primeros.
4. **`ask.py`**: pasa esos 5 fragmentos a un LLM local con Ollama, que responde citándolos.
5. **`evaluate.py`**: mide la calidad del sistema con un conjunto de preguntas de prueba.

## Decisiones

- **Todo en local.** Nada sale del ordenador, no cuesta dinero por consulta y cualquier alumno puede ejecutarlo.
- **Búsqueda híbrida.** La normativa está llena de siglas y nombres propios (CAT, Ebrón, MUCEIM, Turnitin) que BM25 encuentra bien. Los embeddings encuentran las preguntas formuladas con otras palabras. Combinarlas da mejor resultado que cualquiera de las dos por separado (ver la tabla de abajo).
- **Modelos multilingües.** Los documentos están en castellano y alguno en valenciano. `bge-m3` y su reranker funcionan bien con los dos idiomas.
- **Filtro por campus.** Valencia y Alcoy tienen calendarios distintos. Si la pregunta menciona Alcoy, la EPSA, Valencia o la ETSINF, solo se buscan documentos de ese campus y los comunes de la UPV. En el chat también se puede fijar a mano con `/campus`.
- **Saber decir "no lo sé".** Si ningún fragmento supera un umbral mínimo de relevancia, el asistente no llama al LLM y remite a Secretaría. El prompt también le obliga a avisar cuando un dato es de otra escuela o de un curso antiguo.
- **Respuestas deterministas.** El LLM responde con temperatura 0 y sin razonamiento previo, así que es más rápido y la misma pregunta da siempre la misma respuesta.

## Herramientas y hardware

| | |
|---|---|
| Lenguaje | Python 3.12 |
| Lectura de PDF | `pymupdf4llm` |
| Embeddings | `BAAI/bge-m3` (`sentence-transformers`) |
| Búsqueda léxica | `rank-bm25` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | `qwen3.5:4b` con Ollama |
| Hardware de desarrollo | Intel Core i7-12700F, 16 GB de RAM, NVIDIA RTX 3060 |

Los embeddings y el reranker se ejecutan en la CPU. El LLM lo ejecuta Ollama en la GPU. Un modelo de 4B ocupa unos 3,3 GB, así que el proyecto funciona en un portátil normal; sin GPU solo tarda más en responder.

## Resultados

Con 39 preguntas de prueba (de Valencia, de Alcoy, comunes, y 5 que no tienen respuesta en los documentos):

| Recuperación | Hit@5 | MRR |
|---|---|---|
| Solo BM25 | 100% | 0.91 |
| Solo embeddings | 100% | 0.93 |
| Híbrida | 100% | 0.94 |
| Híbrida + reranker | 100% | **0.95** |

En la primera medición de las respuestas, el 82% contenía todos los datos esperados, el 100% citaba alguna fuente y el asistente no se inventó la respuesta en ninguna de las preguntas que no la tienen. Casi todos los fallos se debían a que el modelo escribía la fecha en otro formato.

## Uso

Necesitas Python 3.12 y [Ollama](https://ollama.com).

```bash
python -m venv .RAGenv
.RAGenv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3.5:4b

python ingest.py        # PDF -> data/chunks.jsonl
python index.py         # embeddings -> data/embeddings.npy
python ask.py           # chat en la terminal (/campus alcoy|valencia|todos, /salir)
python ask.py "¿Qué documentos tengo que subir para la defensa?"   # una sola pregunta

python evaluate.py --ablacion     # compara los métodos de búsqueda
python evaluate.py --respuestas   # evalúa también las respuestas del LLM
```

Para añadir documentos, déjalos en `documents/`, añade su escuela, campus y curso al diccionario `META` de `ingest.py` y vuelve a ejecutar `ingest.py` e `index.py`.

## Limitaciones

- Solo sabe lo que dicen los PDF de `documents/`. Si la escuela publica un calendario nuevo, hay que añadirlo.
- Algunos documentos son de 2019 y 2020. El asistente avisa cuando usa uno, pero lo que manda es la normativa vigente.
- Las tablas complejas a veces se leen mal al convertir los PDF a texto.
- No sustituye a Secretaría: para cualquier caso particular, pregunta allí.
