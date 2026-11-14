# Este script lee los documentos de la carpeta documents y guarda los fragmentos en data

import json
import re
from pathlib import Path

import pymupdf4llm

DOCS = Path("documents")
OUT = Path("data/chunks.jsonl")
MAX_CHARS = 1000    # Alrededor de 250 tokens por fragmento

META = {
    "Calendario-de-gestion-TFG-TFM-2026-27-v1.pdf": {"escuela": "EPSA (Alcoy)", "curso": "2026-27"},
    "Fechas-Importantes-TFG-2026-2027.pdf": {"escuela": "ETSINF (Valencia)", "curso": "2026-27"},
    "Guia-dar-Alta-TFG-TFM-abril2019.pdf": {"escuela": "UPV", "curso": "2019"},
    "Manual-Gestion-TFG-TFM-en-movilidad.pdf": {"escuela": "EPSA (Alcoy)", "curso": "2025"},
    "PROCEDIMIENTO-TFG-INTEGRADO-Dobles-Grados-Alumnos.pdf": {"escuela": "UPV", "curso": "s/f"},
    "Preguntas-frecuentes-TFG_TFMcurso26-27.pdf": {"escuela": "ETSINF (Valencia)", "curso": "2026-27"},
    "ProcedimTribunal-y-Defensa-CAT_04_06_2020.pdf": {"escuela": "ETSINF (Valencia)", "curso": "2020"},
    "Procedimiento-para-las-defensas-TFG-TFM-por-videoconferencia-via-Microdoft-TEAMS.pdf": {"escuela": "ETSINF (Valencia)", "curso": "s/f"},
}

def split(text): #Agrupa párrafos hasta MAX_CHARS
    
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) > MAX_CHARS: # si añado el nuevo bloque y supera el max 
            chunks.append(current) # solo guardo hasta el momento
            current = ""
        current = f"{current}\n\n{block}" if current else block #en otro caso lo añado 
    if current:
        chunks.append(current)
    return chunks


def main():
    OUT.parent.mkdir(exist_ok=True) # aseguramos que exista el directorio

    n = 0
    with OUT.open("w", encoding="utf-8") as f:

        for pdf in sorted(DOCS.glob("*.pdf")): # para cada pdf
            pages = pymupdf4llm.to_markdown(str(pdf), page_chunks=True) # libreria para leer pdfs

            for page_num, page in enumerate(pages, start=1):
                page_text = page["text"].replace("<br>", " ")  # saltos dentro de celdas

                for text in split(page_text): #iteramos sobre la lista de textos generada en split()
                    if len(text) < 50:  # numeros de pagina cabeceras sueltas
                        continue
                    record = {
                        "id": f"{pdf.stem}-p{page_num}-{n}",
                        "source": pdf.name,
                        "page": page_num,
                        **META.get(pdf.name, {}),
                        "text": text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} fragmentos -> {OUT}")


if __name__ == "__main__":
    main()