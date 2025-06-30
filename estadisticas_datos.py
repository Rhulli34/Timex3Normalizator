import json
from collections import Counter
import numpy as np

# --- Ruta a tus archivos ---
DETECTION_PATH = "temporal_expressions_dataset_clean.jsonl"
NORMALIZATION_PATH = "micro_dataset_final.json"

# --- Estadísticas para Databank 1 (detección) ---
tokens_lens = []
labels_flat = []

with open(DETECTION_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        toks = ex["tokens"]
        labs = ex["labels"]
        tokens_lens.append(len(toks))
        # Filtramos etiquetas válidas (0 fuera, 1 B, 2 I)
        labels_flat.extend([l for l in labs if l in (1,2)])

avg_tokens = np.mean(tokens_lens)
pct_b = labels_flat.count(1) / len(labels_flat) * 100
pct_i = labels_flat.count(2) / len(labels_flat) * 100

print("=== Databank 1 (detección) ===")
print(f"Longitud media de tokens: {avg_tokens:.1f}")
print(f"% B-TIMEX: {pct_b:.2f}%")
print(f"% I-TIMEX: {pct_i:.2f}%")

# --- Estadísticas para Databank 2 (normalización) ---
expr_lens = []
type_counter = Counter()

with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        expr = ex["expression_original"]
        expr_lens.append(len(expr.split()))
        xml = ex["target_timex3"]
        start = xml.find('type="') + 6
        end = xml.find('"', start)
        tp = xml[start:end] if start>=6 and end>start else "UNKNOWN"
        type_counter[tp] += 1

avg_expr = np.mean(expr_lens)
print("\n=== Databank 2 (normalización) ===")
print(f"Longitud media de expresiones: {avg_expr:.1f} palabras")
print("Distribución por tipo:")
for tp, cnt in type_counter.items():
    print(f"  {tp}: {cnt} ejemplos")