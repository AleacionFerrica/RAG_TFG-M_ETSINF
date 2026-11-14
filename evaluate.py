"""Mide la recuperación (Hit@5, MRR) y, con --respuestas, la calidad de las respuestas.

python evaluate.py               recuperación del sistema completo
python evaluate.py --respuestas  además llama al LLM: acierto, citas y rechazos
python evaluate.py --ablacion    compara los métodos de recuperación (sin LLM)

En preguntas.csv, respuesta_esperada son datos clave separados por ";" que la respuesta debe
contener; las preguntas sin respuesta en el corpus llevan fuente y respuesta vacías.
"""
import csv
import re
import sys
from pathlib import Path

from retrieve import MODES, detect_campus, retrieve, tokenize

QUESTIONS = Path("eval/preguntas.csv")
RESULTS = Path("eval/resultados.csv")


def load_rows() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def retrieval_metrics(rows: list[dict], mode: str = "hybrid+rerank") -> tuple[float, float]:
    """Rellena rango_fuente y score_top en cada fila; devuelve (Hit@5, MRR) sobre las preguntas con fuente."""
    reciprocal_ranks = []
    for row in rows:
        hits = retrieve(row["pregunta"], campus=detect_campus(row["pregunta"]), mode=mode)
        row["score_top"] = f"{hits[0]['score']:.3f}"
        if not row["fuente_esperada"]:
            row["rango_fuente"] = ""
            continue
        sources = [h["source"] for h in hits]
        rank = sources.index(row["fuente_esperada"]) + 1 if row["fuente_esperada"] in sources else None
        row["rango_fuente"] = rank or "-"
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    n = len(reciprocal_ranks)
    return sum(r > 0 for r in reciprocal_ranks) / n, sum(reciprocal_ranks) / n


def contains_all(answer: str, expected: str) -> bool:
    """Cada dato clave (separados por ";") debe aparecer como palabras completas; "a|b" acepta variantes."""
    text = f" {' '.join(tokenize(answer))} "
    return all(
        any(f" {' '.join(tokenize(alt))} " in text for alt in key.split("|"))
        for key in expected.split(";") if key.strip()
    )


assert contains_all("Del 09/12/2026 al 11/12/2026 [1]", "09/12/2026;11/12/2026")
assert not contains_all("según la normativa", "no")
assert contains_all("Son tres miembros", "3|tres")


def answer_metrics(rows: list[dict]) -> None:
    from ask import NO_ANSWER, ask  # solo se carga el LLM si se piden respuestas

    for row in rows:
        answer = ask(row["pregunta"])[0]
        row["respuesta"] = answer
        row["cita"] = bool(re.search(r"\[\d+\]", answer))
        if row["fuente_esperada"]:
            row["acierto"] = contains_all(answer, row["respuesta_esperada"])
        else:
            row["rechazo_ok"] = answer == NO_ANSWER or "secretaria" in tokenize(answer)

    answerable = [r for r in rows if r["fuente_esperada"]]
    unanswerable = [r for r in rows if not r["fuente_esperada"]]
    print(f"Acierto:  {sum(r['acierto'] for r in answerable) / len(answerable):.0%}  (contiene todos los datos clave)")
    print(f"Citas:    {sum(r['cita'] for r in answerable) / len(answerable):.0%}  (cita algún fragmento [n])")
    if unanswerable:
        print(f"Rechazos: {sum(r['rechazo_ok'] for r in unanswerable) / len(unanswerable):.0%}  (no inventa cuando no hay respuesta)")


def main() -> None:
    rows = load_rows()

    if "--ablacion" in sys.argv:
        print(f"{'modo':<15} Hit@5   MRR")
        for mode in MODES:
            hit, mrr = retrieval_metrics(rows, mode)
            print(f"{mode:<15} {hit:>5.0%}  {mrr:.2f}")
        return

    hit, mrr = retrieval_metrics(rows)
    print(f"Preguntas: {len(rows)}")
    print(f"Hit@5: {hit:.0%}  (la fuente correcta está entre los 5 fragmentos)")
    print(f"MRR:   {mrr:.2f}  (1.0 = siempre en primera posición)")
    if "--respuestas" in sys.argv:
        answer_metrics(rows)

    fields = list(dict.fromkeys(k for row in rows for k in row))  # unión de columnas, en orden
    with RESULTS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detalle por pregunta -> {RESULTS}")


if __name__ == "__main__":
    main()
