En esta carpeta se encuentra el codigo que se ha usado para crear el space en huggingface.


# Reconocimiento y Normalización de Expresiones Temporales (TIMEX3)-

**Demo online** para detectar y normalizar expresiones temporales en textos, usando modelos de lenguaje (LLMs) personalizados.

## 🚀 ¿Qué hace este Space?

* **Reconoce expresiones temporales** en texto libre o archivos (.txt y .pdf).
* **Normaliza las expresiones** al estándar [TIMEX3](https://timeml.org/site/publications/timeMLdocs/timex3-technical-spec.pdf).
* Permite **descargar los resultados** como archivo CSV.
* Procesamiento rápido y seguro (limite: 8.000 caracteres por entrada).

## 🧑‍💻 ¿Cómo usarlo?

1. **Introduce un texto** en la pestaña “Texto” o **sube un archivo** `.txt` o `.pdf` en la pestaña “Archivo”.
2. Selecciona la fecha de refenrencia del texto
3. Haz clic en **“Procesar”**.
4. Visualiza el resultado en una tabla con:

   * Columna 1: Expresión temporal detectada
   * Columna 2: Normalización TIMEX3
5. Descarga el resultado como CSV si lo deseas.

## 🛠️ Modelos usados

* **NER temporal**: [`Rhulli/Roberta-ner-temporal-expresions-secondtrain`](https://huggingface.co/Rhulli/Roberta-ner-temporal-expresions-secondtrain)
* **Normalizador TIMEX3**: [`Rhulli/gemma-2b-it-TIMEX3`](https://huggingface.co/Rhulli/gemma-2b-it-TIMEX3)

## 📄 Ejemplo de uso

Texto de entrada:

> "El concierto es el 14 de julio de 2025. La garantía dura dos años."

| Expresión temporal  | Normalización TIMEX3                                                  |
| ------------------- | --------------------------------------------------------------------- |
| 14 de julio de 2025 | \[TIMEX3 type='DATE' value='2025-07-14']14 de julio de 2025\[/TIMEX3] |
| dos años            | \[TIMEX3 type='DURATION' value='P2Y']dos años\[/TIMEX3]               |

## 📦 Requisitos técnicos

* [Gradio](https://gradio.app/)
* [Transformers](https://huggingface.co/docs/transformers/index)
* [PyPDF2](https://pypdf2.readthedocs.io/en/latest/)
* [Torch](https://pytorch.org/)

(Ver `requirements.txt` para detalles)

## ⚠️ Notas

* No subas información sensible. Los archivos y textos enviados pueden ser procesados temporalmente en el servidor.
* Límite de texto: 8.000 caracteres por entrada.
* Idioma recomendado, Inglés.

## 👨‍🎓 Autor

Desarrollado por Raúl Moreno Mejías como parte de un proyecto de investigación en reconocimiento y normalización de expresiones temporales con LLMs.

---


