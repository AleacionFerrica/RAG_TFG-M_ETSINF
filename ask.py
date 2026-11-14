"""Responde una pregunta con un LLM local (Ollama) usando solo los fragmentos recuperados."""
import sys

import ollama

from retrieve import retrieve

MODEL = "qwen3.5:4b"  # cualquier modelo descargado con `ollama pull`

SYSTEM = """Eres el asistente de normativa de TFG/TFM de la ETSINF (Universitat Politècnica de València).
Responde en castellano usando SOLO la información de los fragmentos que se te dan.
- Cita cada dato con el número de su fragmento entre corchetes, por ejemplo [2].
- Si los fragmentos no contienen la respuesta, dilo claramente y sugiere consultar a Secretaría de la ETSINF. No inventes fechas ni plazos.
- Si un fragmento es de otra escuela (EPSA Alcoy) o de un curso anterior, avísalo antes de usar ese dato.
- Responde en pocas frases, sin repetir la pregunta."""


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] {h['source']}, pág. {h['page']} (escuela: {h.get('escuela', '?')}, curso: {h.get('curso', '?')})\n{h['text']}"
        for n, h in enumerate(hits, start=1)
    )


def ask(question: str) -> tuple[str, list[dict]]:
    hits = retrieve(question)
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Fragmentos:\n\n{format_context(hits)}\n\nPregunta: {question}"},
        ],
        think=False,  # respuesta directa, sin razonamiento previo: más rápido
        options={"temperature": 0, "num_ctx": 8192},  # determinista; cabe pregunta + 5 fragmentos
    )
    return response.message.content or "", hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "¿Cuándo puedo solicitar la defensa si soy del MUCEIM en Alcoy?"
    answer, hits = ask(question)
    print(answer, "\n\nFuentes:")
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h['source']}, pág. {h['page']}")