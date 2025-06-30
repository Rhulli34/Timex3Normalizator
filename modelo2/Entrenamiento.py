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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login
import os

# Soluciona el problema de memoria OutOfMemory (opcional pero recomendado)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Iniciando sesión en Hugging Face...")
login()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --- Definición de la función con el formato de prompt para GEMMA ---
def create_gemma_prompt_function(example):
    """Crea el prompt formateado para el entrenamiento con el formato de Gemma."""
    prompt_pregunta = f"""Tu tarea es normalizar la expresión temporal al formato TIMEX3, utilizando la fecha de anclaje (DCT) cuando sea necesario.

Fecha de Anclaje (DCT): {example["dct"]}
Expresión Original: {example["expression_original"]}"""

    respuesta_modelo = example["target_timex3"]

    # Plantilla específica de Gemma
    prompt_template = f"""<start_of_turn>user
{prompt_pregunta}<end_of_turn>
<start_of_turn>model
{respuesta_modelo}<end_of_turn>"""

    return {"text": prompt_template}


print("Cargando y procesando dataset...")
full_dataset = load_dataset('json', data_files="/content/micro_dataset_final.json", split='train')

# Dividimos en entrenamiento (90%) y evaluación (10%)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

processed_train_dataset = split_dataset['train'].map(create_gemma_prompt_function)
processed_eval_dataset = split_dataset['test'].map(create_gemma_prompt_function)

print(f"✅ Datos de entrenamiento: {len(processed_train_dataset)}")
print(f"✅ Datos de evaluación: {len(processed_eval_dataset)}")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ===================================================================
# CELDA MAESTRA DE EVALUACIÓN
# ===================================================================

import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score

def extract_value(text):
    """Extrae el atributo 'value' de una etiqueta TIMEX3."""
    match = re.search(r'value="([^"]+)"', text)
    if match:
        return match.group(1)
    return ""

def compute_metrics(eval_pred):
    """
    Función que el Trainer llamará para calcular nuestras métricas (F1, etc.).
    """
    # Importamos numpy DENTRO para máxima robustez
    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    predictions[predictions == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    parsed_preds = [extract_value(pred) for pred in decoded_preds]
    parsed_labels = [extract_value(label) for label in decoded_labels]

    y_true = []
    y_pred = []

    for pred, label in zip(parsed_preds, parsed_labels):
        if label != "":
            y_true.append(1)
            y_pred.append(1 if pred == label else 0)

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}

print("✅ Funciones para la evaluación definidas correctamente.")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Carga del Modelo ---
model_id = "google/gemma-2b-it"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 only FP16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# --- Prepara LoRA ---
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.02,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- Argumentos de entrenamiento ---
training_args = TrainingArguments(
    output_dir="./results_gemma_timex3",
    seed=42,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,                # 10% warmup
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,

    fp16=True,
    optim="paged_adamw_8bit",

    num_train_epochs=10,              # más épocas
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,

    logging_strategy="steps",
    logging_steps=200,
    report_to="none",
)


# --- Data collator para padding dinámico ---
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- Trainer con callbacks y métricas ---
from transformers import EarlyStoppingCallback
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

print("✅ Trainer configurado con mejoras para reproducibilidad, monitoreo y parada temprana.")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("🚀 ¡Iniciando el entrenamiento de Gemma!")
trainer.train()
print("✅ ¡Entrenamiento completado!")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Subir modelo a Hugging Face

# 2) Imports y configuración de Quantization + LoRA igual que en tu entrenamiento
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, PeftModel
from huggingface_hub import HfApi, Repository

# Ajusta estos parámetros a los que usaste en el entrenamiento
model_id = "google/gemma-2b-it"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 usa fp16
)

# 3) Define tu usuario y repo en HF
HF_USER = "Rhulli"                        # <— tu usuario HF
HF_REPO  = "gemma-2b-it-TIMEX3"        # <— nombre de tu repo
HF_REPO_ID = f"{HF_USER}/{HF_REPO}"

# 4) Crea el repo en Hugging Face si no existe
hf_api = HfApi()
hf_api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)

# 5) Clona el repo localmente
repo = Repository(
    local_dir="hf_repo",
    clone_from=HF_REPO_ID,
    use_auth_token=True
)

# 6) Carga el modelo base + adaptadores LoRA desde tu checkpoint
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
base = prepare_model_for_kbit_training(base)
model = PeftModel.from_pretrained(base, "./results_gemma_timex3/checkpoint-176")

# 7) Tokenizer (igual que en tu entrenamiento)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 8) Guarda en la carpeta clonada y push
model.save_pretrained("hf_repo")
tokenizer.save_pretrained("hf_repo")
repo.push_to_hub(commit_message="Upload LoRA-finetuned gemma-2b-it")

print(f"✅ Modelo subido: https://huggingface.co/{HF_REPO_ID}")
