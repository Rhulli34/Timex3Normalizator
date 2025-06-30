import os
import re
import unicodedata
import io
import torch
import gradio as gr
import pdfplumber
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --- Funciones de normalización y limpieza ---
_SPACE_VARIANTS = r"[\u202f\u00a0\u2009\u200a\u2060]"

def _normalise_apostrophes(text: str) -> str:
    return text.replace("´", "'").replace("’", "'")

def _normalise_spaces(text: str, collapse: bool = True) -> str:
    text = re.sub(_SPACE_VARIANTS, " ", text)
    text = unicodedata.normalize("NFKC", text)
    if collapse:
        text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def _clean_timex(ent: str) -> str:
    ent = ent.replace("</s>", "").strip()
    return re.sub(r"[\.]+$", "", ent)

# --- Identificadores de los modelos ---
NER_ID      = "Rhulli/Roberta-ner-temporal-expresions-secondtrain"
ID2LABEL    = {0: "O", 1: "B-TIMEX", 2: "I-TIMEX"}
BASE_ID     = "google/gemma-2b-it"
ADAPTER_ID  = "Rhulli/gemma-2b-it-TIMEX3"

# --- Configuración de cuantización para el modelo de normalización ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- Leer el token del entorno (añadido como Repository Secret) ---
HF_TOKEN = os.getenv("HF_TOKEN")

def load_models():
    ner_tok = AutoTokenizer.from_pretrained(NER_ID, token=HF_TOKEN)
    ner_mod = AutoModelForTokenClassification.from_pretrained(NER_ID, token=HF_TOKEN)
    ner_mod.eval()
    if torch.cuda.is_available():
        ner_mod.to("cuda")

    base_mod = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        device_map="auto",
        token=HF_TOKEN
    )
    norm_tok = AutoTokenizer.from_pretrained(ADAPTER_ID, use_fast=True, token=HF_TOKEN)
    norm_mod = PeftModel.from_pretrained(
        base_mod,
        ADAPTER_ID,
        device_map="auto",
        token=HF_TOKEN
    )
    norm_mod.eval()

    return ner_tok, ner_mod, norm_tok, norm_mod

# Carga inicial de los modelos
ner_tok, ner_mod, norm_tok, norm_mod = load_models()
eos_id = norm_tok.convert_tokens_to_ids("<end_of_turn>")

# --- Lectura de archivos ---
def read_file(file_obj) -> str:
    path = file_obj.name
    if path.lower().endswith('.pdf'):
        full = ''
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    full += txt + '\n'
        return full
    else:
        with open(path, 'rb') as f:
            data = f.read()
        try:
            return data.decode('utf-8')
        except:
            return data.decode('latin-1', errors='ignore')

# --- Procesamiento de texto ---
def extract_timex(text: str):
    text_norm = _normalise_spaces(_normalise_apostrophes(text))
    inputs = ner_tok(text_norm, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = ner_mod(**inputs).logits

    preds  = torch.argmax(logits, dim=2)[0].cpu().numpy()
    tokens = ner_tok.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current  = []
    for tok, lab in zip(tokens, preds):
        tag = ID2LABEL.get(lab, "O")
        if tag == "B-TIMEX":
            if current:
                entities.append(ner_tok.convert_tokens_to_string(current).strip())
            current = [tok]
        elif tag == "I-TIMEX" and current:
            current.append(tok)
        else:
            if current:
                entities.append(ner_tok.convert_tokens_to_string(current).strip())
                current = []
    if current:
        entities.append(ner_tok.convert_tokens_to_string(current).strip())

    return [_clean_timex(e) for e in entities]

def normalize_timex(expr: str, dct: str) -> str:
    prompt = (
        f"<start_of_turn>user\n"
        f"Tu tarea es normalizar la expresión temporal al formato TIMEX3, utilizando la fecha de anclaje (DCT) cuando sea necesaria.\n"
        f"Fecha de Anclaje (DCT): {dct}\n"
        f"Expresión Original: {expr}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs  = norm_tok(prompt, return_tensors="pt").to(norm_mod.device)
    outputs = norm_mod.generate(**inputs, max_new_tokens=64, eos_token_id=eos_id)

    full_decoded = norm_tok.decode(
        outputs[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=False
    )
    raw_tag  = full_decoded.split("<end_of_turn>")[0].strip()
    return raw_tag.replace("[", "<").replace("]", ">")

# --- Pipeline principal ---
def run_pipeline(files, raw_text, dct):
    rows = []
    file_list = files if isinstance(files, list) else ([files] if files else [])

    if raw_text:
        for line in raw_text.splitlines():
            if line.strip():
                for expr in extract_timex(line):
                    rows.append({
                        'Expresión': expr,
                        'Normalización': normalize_timex(expr, dct)
                    })

    for f in file_list:
        content = read_file(f)
        for line in content.splitlines():
            if line.strip():
                for expr in extract_timex(line):
                    rows.append({
                        'Expresión': expr,
                        'Normalización': normalize_timex(expr, dct)
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([], columns=['Expresión', 'Normalización'])

    return df, ""

# --- Interfaz Gradio ---
with gr.Blocks() as demo:
    gr.Markdown(
        ## TIMEX Extractor & Normalizer
        """"
        Esta aplicación permite extraer expresiones temporales de textos o archivos (.txt, .pdf)
        y normalizarlas a formato TIMEX3.

        **Cómo usar:**
        - Sube uno o varios archivos en la columna izquierda.
        - Ajusta la *Fecha de Anclaje (DCT)* justo debajo de los archivos.
        - Escribe o pega tu texto en la columna derecha.
        - Pulsa **Procesar** para ver los resultados en la tabla debajo.

        **Columnas de salida:**
        - *Expresión*: la frase temporal extraída.
        - *Normalización*: la etiqueta TIMEX3 generada.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            files     = gr.File(file_types=['.txt', '.pdf'], file_count='multiple', label='Archivos (.txt, .pdf)')
            dct_input = gr.Textbox(value="2025-06-11", label="Fecha de Anclaje (YYYY-MM-DD)")
            run_btn   = gr.Button("Procesar")
        with gr.Column(scale=2):
            raw_text  = gr.Textbox(lines=15, placeholder='Pega o escribe aquí tu texto...', label='Texto libre')

    output_table = gr.Dataframe(headers=['Expresión', 'Normalización'], label="Resultados", type="pandas")
    output_logs  = gr.Textbox(label="Logs", lines=5, interactive=False)

    # Después de definir output_table y output_logs:
    download_btn      = gr.Button("Descargar CSV")
    csv_file_output  = gr.File(label="Descargar resultados en CSV", visible=False)

    # El click de procesar normales
    run_btn.click(
        fn=run_pipeline,
        inputs=[files, raw_text, dct_input],
        outputs=[output_table, output_logs]
    )

    # Función para exportar a CSV
    def export_csv(df):
        csv_path = "resultados.csv"
        df.to_csv(csv_path, index=False)
        return gr.update(value=csv_path, visible=True)

    # Asociar el botón de descarga al CSV
    download_btn.click(
        fn=export_csv,
        inputs=[output_table],
        outputs=[csv_file_output]
    )

    # Lanzar la app
    demo.launch()

