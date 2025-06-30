#Instalamos Librerias

pip install transformers
pip install --upgrade datasets
pip install seqeval
pip install evatalue

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Subimos dataset(COLAB)

from google.colab import files

uploaded = files.upload()

# Esto te permite verificar el nombre del archivo subido
for filename in uploaded.keys():
    print(f'Archivo subido: {filename}')


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Tratamos el Dataset

from datasets import Dataset
import random
import numpy as np
import torch

# Configuración global de semillas para reproducibilidad
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 1. Cargar el dataset desde el archivo JSONL
dataset = Dataset.from_json("temporal_expressions_dataset_clean.jsonl")

# 2. Mezclar y dividir en entrenamiento (80%) y test (20%)
dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.2)

# 3. Definir los mapeos de etiquetas
label_list = ["O", "B-TIMEX", "I-TIMEX"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# 4. Vista previa de un ejemplo
print("Ejemplo de training:")
print(dataset["train"][200])

# Tamaño final del dataset
print(f'Tamaño train: {len(dataset["train"])} - Tamaño test: {len(dataset["test"])}')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Tokenizar los textos y alinear las etiquetas

from transformers import AutoTokenizer

# Cargar el tokenizer del modelo que usarás (ajusta el nombre si usas otro)
model_checkpoint = "Rhulli/Roberta-ner-temporal-expresions-2.0"  # o el que tú estás usando
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Obtener la lista de etiquetas y funciones de mapeo
label_list = ["O", "B-TIMEX", "I-TIMEX"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Función para tokenizar y alinear las etiquetas
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)

    labels = []
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Para tokens especiales como [CLS], [SEP]
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])  # Primer subtoken
        else:
            labels.append(-100)  # Subtoken interno, lo ignoramos
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Aplicar la función al dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=False)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Cargar el modelo para token classification

from transformers import AutoModelForTokenClassification
import torch

model_checkpoint = "Rhulli/Roberta-ner-temporal-expresions-2.0" #Modelo que quieras entrenar

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Comprobar dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Usando dispositivo: {device}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Definir las métricas

import numpy as np
import evaluate

# Cargar métrica
seqeval = evaluate.load("seqeval")

# Convertimos IDs de etiquetas a sus nombres reales
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Configurar argumentos de entrenamiento y Crear el Trainer

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

# 1) Asegura GPU
assert torch.cuda.is_available(), "No hay GPU disponible"
device = torch.device("cuda")
model.to(device)

# 2) Reduce batch y activa fp16
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,   # reducido
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,                       # Mixed precision
    seed=42,
)

# 3) Habilita gradient checkpointing
model.gradient_checkpointing_enable()

# 4) Limpia cache
torch.cuda.empty_cache()

# 5) Crea Trainer normalmente
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 6) Entrena
trainer.train()
trainer.evaluate()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Guardar Modelo

# Guardar modelo entrenado
trainer.save_model("temporal-entity-model")

# Guardar tokenizer (¡también es importante!)
tokenizer.save_pretrained("temporal-entity-model")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Descargar el Modelo 

# 1. Comprime la carpeta del modelo
zip -r temporal-entity-model.zip temporal-entity-model

# 2. Usa la utilidad de Colab para descargar
from google.colab import files
files.download("temporal-entity-model.zip")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

