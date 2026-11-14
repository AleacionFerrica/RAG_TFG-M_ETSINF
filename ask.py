"""Responde una pregunta con un LLM local (Ollama) usando solo los fragmentos recuperados."""
import sys

import ollama

from retrieve import detect_campus, retrieve

MODEL = "qwen3.5:4b"  # cualquier modelo descargado con `ollama pull`
MIN_SCORE = 0.05  # calibrado con eval (score_top): la pregunta válida más baja da 0.116. Por debajo, ningún fragmento es relevante: se responde NO_ANSWER sin llamar al LLM
NO_ANSWER = "No encuentro esa información en la normativa de TFG/TFM disponible. Consulta a la Secretaría de tu escuela."

SYSTEM = """Eres el asistente de normativa de TFG/TFM de la Universitat Politècnica de València para dos escuelas: ETSINF (Valencia) y EPSA (Alcoy).
Responde en castellano usando SOLO la información de los fragmentos que se te dan.
- Cita cada dato con el número de su fragmento entre corchetes, por ejemplo [2].
- Si los fragmentos no contienen la respuesta, dilo claramente y sugiere consultar a la Secretaría de la escuela. No inventes fechas ni plazos.
- Si la pregunta es de una escuela y el fragmento es de la otra, o de un curso anterior, avísalo antes de usar ese dato.
- Responde en pocas frases, sin repetir la pregunta."""


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] {h['source']}, pág. {h['page']} (escuela: {h.get('escuela', '?')}, curso: {h.get('curso', '?')})\n{h['text']}"
        for n, h in enumerate(hits, start=1)
    )


def ask(question: str, campus: str | None = None) -> tuple[str, list[dict]]:
    hits = retrieve(question, campus=campus or detect_campus(question))
    if hits[0]["score"] < MIN_SCORE:
        return NO_ANSWER, []
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


def answer_and_print(question: str, campus: str | None) -> None:
    if not campus and detect_campus(question):
        print(f"(campus detectado: {detect_campus(question)})")
    answer, hits = ask(question, campus)
    print(answer, "\n\nFuentes:")
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h['source']}, pág. {h['page']} ({h.get('campus', '?')})")


def chat() -> None:
    """Bucle interactivo: los modelos se cargan una vez y se reutilizan en cada pregunta."""
    campus = None
    print("Pregunta lo que quieras. Comandos: /campus alcoy | valencia | todos, /salir")
    while True:
        try:
            line = input(f"\n[{campus or 'todos'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/salir", "/exit", "/q"):
            break
        if line.startswith("/campus"):
            arg = line.removeprefix("/campus").strip().capitalize()
            campus = arg if arg in ("Alcoy", "Valencia") else None
            print(f"Campus: {campus or 'todos'}")
        elif line:
            answer_and_print(line, campus)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer_and_print(" ".join(sys.argv[1:]), None)
    else:
        chat()