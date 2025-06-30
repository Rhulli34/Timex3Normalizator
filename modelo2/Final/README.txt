En esta carpeta se encuentra lo que se ha necesitado para entrenar el segundo modelo.

micro_dataset_final el cual ha sido usado para entrenar el modelo

El modelo que ha sido entrenado es google/gemma-2-2b-it (https://huggingface.co/google/gemma-2-2b-it)

El modelo obtenido es Rhulli/gemma-2b-it-TIMEX3 (https://huggingface.co/Rhulli/gemma-2b-it-TIMEX3) y estos son sus scores en su major epoch:

Training Loss	Validation Loss		Precision	Recall		F1

0.635200	 0.185600		1.000000	0.700000	0.823529


Se deja el codigo que se ha usado tanto para entrenarlo como para probarlo.

(El codigo ha sido usado en colab)