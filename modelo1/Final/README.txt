En esta carpeta se encuentra lo que se ha necesitado para entrenar el priemer modelo.

dataset_inicial el cual se uso para un primer entrenamiento al modelo DAMO-NLP-SG/roberta-time_identification (https://huggingface.co/DAMO-NLP-SG/roberta-time_identification)

Tras este entrenamiento se subio el modelo Rhulli/Roberta-ner-temporal-expresions-2.0 (https://huggingface.co/Rhulli/Roberta-ner-temporal-expresions-2.0)

El modelo final ha sido creado entrenando el modelo Roberta-ner-temporal-expresions-2.0, con un dataset mas refinado que une el antiguo dataset y mejoras considerable en el aparte de ser mas grande.

Modelo final es Rhulli/Roberta-ner-temporal-expresions-secondtrain y estos son sus scores

{'eval_loss': 0.14976459741592407,
 'eval_precision': 0.8782051282051282,
 'eval_recall': 0.8782051282051282,
 'eval_f1': 0.8782051282051282,
 'eval_accuracy': 0.9680327868852459,
 'eval_runtime': 0.4758,
 'eval_samples_per_second': 231.202,
 'eval_steps_per_second': 29.426,
 'epoch': 3.0}

Se añade el codigo con el que se ha entrado y un script para probar el modelo de forma individual. 

El codigo verificar_dataset sirve para ver que estan bien alineados los labels y token de los dataset, ya que ha sido realizado a mano.

(El codigo ha sido usado en colab)
