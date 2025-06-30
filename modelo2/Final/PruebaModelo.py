# Celda 1: Instalación Definitiva (con NumPy v1.x)

pip install --force-reinstall --no-cache-dir \
    "numpy==1.26.4" \
    "transformers==4.41.2" \
    "accelerate==0.30.1" \
    "peft==0.11.1" \
    "trl==0.8.6" \
    "bitsandbytes==0.43.1" \
    "datasets==2.19.1" \
    "sentencepiece" \
    "triton==2.2.0" \
    "scikit-learn"

from huggingface_hub import login
login()

# ===================================================================
# SCRIPT DE PRUEBA FINAL (Salida Limpia con extracción robusta)
# ===================================================================

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore")

# 1) IDs de tu modelo
base_model_id   = "google/gemma-2b-it"
adapter_repo_id = "Rhulli/gemma-2b-it-finetuned"

# 2) Configuración 4-bit + FP16 (Tesla T4)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 3) Carga condicional del modelo y tokenizer
if 'model' in locals() and 'tokenizer' in locals():
    print("✅ Usando modelo y tokenizer ya en memoria.\n")
else:
    print(f"Cargando modelo base cuantizado: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    print(f"Cargando tokenizer desde: {adapter_repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo_id, use_fast=True)
    print("Aplicando adaptadores LoRA...")
    model = PeftModel.from_pretrained(base_model, adapter_repo_id, device_map="auto")
    model.eval()
    print("✅ Modelo y tokenizer cargados.\n")

# --- 1. Define la fecha y las expresiones a probar ---
# Todas las pruebas usarán esta misma fecha de anclaje
dct_para_todas_las_pruebas = "2025-06-11"

# Lista de expresiones que quieres normalizar
expresiones_a_probar = [
    "tomorrow", "yesterday", "in two days", "next Friday", "last Monday",
    "this afternoon", "two weeks ago", "next year", "three months from now",
    "later this evening", "at 5 PM", "this weekend", "end of next month",
    "the day after tomorrow", "following Wednesday",
    "March 15, 2025", "2023-07-04", "08/05/2021", "2025-12", "2021-08-05T14:30",
    "2023-W49", "2021-Q3", "April 2022", "2025-06-11", "14:30:00",
    "2024-05-23T00:00Z", "2019-08-05", "10:15","every Monday","daily","annually","for three years"
]

# --- 2. Obtenemos el ID del token de parada de Gemma ---
# Esto es crucial para que el modelo sepa cuándo callarse.
eos_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

# --- 3. Bucle para procesar cada expresión ---
print(f"\n--- INICIANDO PRUEBAS (DCT para todas: {dct_para_todas_las_pruebas}) ---")

if 'model' in locals():
    for expression in expresiones_a_probar:
        # Construir el prompt
        prompt_pregunta = f"""Tu tarea es normalizar la expresión temporal al formato TIMEX3, utilizando la fecha de anclaje (DCT) cuando sea necesario.

Fecha de Anclaje (DCT): {dct_para_todas_las_pruebas}
Expresión Original: {expression}"""

        prompt = f"""<start_of_turn>user\n{prompt_pregunta}<end_of_turn>\n<start_of_turn>model\n"""

        # Generar la respuesta
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # La clave está aquí: le pasamos 'eos_token_id' para que pare automáticamente
        outputs = model.generate(**inputs, max_new_tokens=64, eos_token_id=eos_id)

        # Decodificar y aislar la respuesta de forma robusta
        input_length = inputs.input_ids.shape[1]
        response_tokens = outputs[0, input_length:]
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        # Limpiamos la respuesta quedándonos solo con el texto antes del token de parada
        clean_prediction = decoded_response.split("<end_of_turn>")[0].strip()

        # Mostrar el resultado en el formato que pediste
        print(f"Expresión Original: {expression}")
        print(f"Normalización: {clean_prediction}\n")