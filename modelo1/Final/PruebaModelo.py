!pip install transformers
!pip install --upgrade datasets
!pip install seqeval
!pip install evaluate

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import unicodedata          # << añadido

# --- 1 · Paths & labels ------------------------------------------------------
model_path = "Rhulli/Roberta-ner-temporal-expresions-2.0"      # directorio donde guardaste modelo+tokenizer
id2label   = {0: "O", 1: "B-TIMEX", 2: "I-TIMEX"}

# --- 2 · Carga ---------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# --- 3 · Normalización de apóstrofes ----------------------------------------
def _normalise_apostrophes(text: str) -> str:
    """Reemplaza ´ y ’ por ' para que coincida con el special‑token o'clock."""
    return text.replace("´", "'").replace("’", "'")

# --- 3 bis · Normalización de espacios Unicode “raros” ----------------------
_SPACE_VARIANTS = r"[\u202f\u00a0\u2009\u200a\u2060]"   # NNBSP, NBSP, thin, hair, word‑joiner…

def _normalise_spaces(text: str, collapse: bool = True) -> str:
    """
    Sustituye espacios exóticos por U+0020 y, opcionalmente,
    colapsa secuencias de más de un espacio.
    """
    text = re.sub(_SPACE_VARIANTS, " ", text)
    text = unicodedata.normalize("NFKC", text)
    if collapse:
        text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

# --- 3 ter · Limpieza de entidades devueltas --------------------------------
def _clean_timex(ent: str) -> str:
    """Elimina '</s>' y el punto final pegado, si lo hubiese."""
    ent = ent.replace("</s>", "").strip()
    ent = re.sub(r"[\.]+$", "", ent)
    return ent

# --- 4 · Función de inferencia ----------------------------------------------
def extract_timex(text: str):
    text = _normalise_spaces(_normalise_apostrophes(text))

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    preds  = torch.argmax(logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities, current = [], []
    for tok, lab_id in zip(tokens, preds):
        lab = id2label.get(lab_id, "O")
        if lab == "B-TIMEX":
            if current:
                entities.append(tokenizer.convert_tokens_to_string(current).strip())
            current = [tok]
        elif lab == "I-TIMEX" and current:
            current.append(tok)
        else:
            if current:
                entities.append(tokenizer.convert_tokens_to_string(current).strip())
                current = []
    if current:
        entities.append(tokenizer.convert_tokens_to_string(current).strip())

    # Limpieza final
    return [_clean_timex(e) for e in entities]

# --- 5 · Ejemplo -------------------------------------------------------------
frases = [
    "I’ll meet you tomorrow morning at 9 o’clock",
    "She submitted the report last Friday",
    "Our flight leaves on June 5th, 2025.",
    "We’ve been working on this project since January",
    "By the end of next week, we should have the results.",
    "The meeting will be next Thursday at 3:45 and will last two hours.",
    "The conference starts on 12 July 2025 at 09:30 AM.",
    "Let’s reschedule the meeting to next Wednesday afternoon.",
    "She lived in Paris from 1998 to 2002",
    "We’ll need the first draft within three weeks",
    "Production doubled during the second quarter of 2023",
    "Please remind me in 20 minutes to stretch",
    "I’ve been awake since 4 am.",
    "In the 1990s, mobile phones became mainstream",
    "Every Friday at 5 pm we send the weekly report"
]

for text in frases:
    print("Texto:", text)
    print("Expresiones temporales:", extract_timex(text))
    print("-" * 40)
