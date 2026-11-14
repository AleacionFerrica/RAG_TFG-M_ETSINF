"""Mide la recuperación (Hit@5, MRR) y guarda las respuestas para revisarlas."""
import csv
import sys
from pathlib import Path

from retrieve import retrieve

QUESTIONS = Path("eval/preguntas.csv")
RESULTS = Path("eval/resultados.csv")


def main(with_answers: bool) -> None:
    QUESTIONS.parent.mkdir(exist_ok=True) 
    with QUESTIONS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    hits_at_5, reciprocal_ranks = 0, []
    for row in rows:
        sources = [h["source"] for h in retrieve(row["pregunta"])]
        rank = sources.index(row["fuente_esperada"]) + 1 if row["fuente_esperada"] in sources else None
        hits_at_5 += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        row["rango_fuente"] = rank or "-"
        if with_answers:
            from ask import ask  # solo se carga el LLM si se piden respuestas
            row["respuesta"] = ask(row["pregunta"])[0]

    n = len(rows)
    print(f"Preguntas: {n}")
    print(f"Hit@5: {hits_at_5 / n:.0%}  (la fuente correcta está entre los 5 fragmentos)")
    print(f"MRR:   {sum(reciprocal_ranks) / n:.2f}  (1.0 = siempre en primera posición)")

    with RESULTS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detalle por pregunta -> {RESULTS}")


if __name__ == "__main__":
    main(with_answers="--respuestas" in sys.argv)