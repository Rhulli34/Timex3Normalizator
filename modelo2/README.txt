Se intento entrenar el modelo T5 pero no razonaba la fecha

Se creo un dataset micro_dataset_final para usar este razonamiento

LLama y Gemma dos no soportaban el entrenamienot le GPU

Se usa Gemma con este dataset (No temenos el f1) google/gemma-2b-it

Epoch	Training Loss	Validation Loss	Precision	Recall	F1
1	No log	1.022442	1.000000	0.200000	0.333333
2	No log	0.474007	1.000000	0.525000	0.688525
3	No log	0.353309	1.000000	0.550000	0.709677
4	No log	0.279153	1.000000	0.600000	0.750000
5	No log	0.246459	1.000000	0.650000	0.787879
6	No log	0.216441	1.000000	0.650000	0.787879
7	No log	0.191740	1.000000	0.675000	0.805970
8	No log	0.188013	1.000000	0.700000	0.823529
9	No log	0.183922	1.000000	0.700000	0.823529
10	0.635200	0.185600	1.000000	0.700000	0.823529

