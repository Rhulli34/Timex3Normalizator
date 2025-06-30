Se entreno DAMO-NLP-SG/roberta-time_identification con el dataset_inicial. --> Rhulli/Roberta-ner-temporal-expresions-2.0
Problema con ambiguedades y texto largos
Se crean dos datset mas para solucinar este problema temporal_expression_negative_sampling y temporal_expressions_entrenamiento_largo
Se entrena Rhulli/Roberta-ner-temporal-expresions-2.0 con temporal_expressions_dataset_clean
obtenemos {'eval_loss': 0.14976459741592407,
 'eval_precision': 0.8782051282051282,
 'eval_recall': 0.8782051282051282,
 'eval_f1': 0.8782051282051282,
 'eval_accuracy': 0.9680327868852459,
 'eval_runtime': 0.4758,
 'eval_samples_per_second': 231.202,
 'eval_steps_per_second': 29.426,
 'epoch': 3.0}
---> Rhulli/Roberta-ner-temporal-expresions-secondtrain