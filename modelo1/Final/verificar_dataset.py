import json

def verificar_alineacion_tokens_labels(ruta_archivo):
    """
    Verifica que la cantidad de tokens coincida con la cantidad de etiquetas
    para cada entrada en un archivo JSONL.

    Args:
        ruta_archivo (str): La ruta al archivo JSONL.
    """
    errores_encontrados = 0
    lineas_procesadas = 0

    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for numero_linea, linea_texto in enumerate(f, 1):
                lineas_procesadas += 1
                try:
                    # Cargar el JSON de la línea actual
                    dato = json.loads(linea_texto.strip())

                    # Verificar la presencia de las claves 'tokens' y 'labels'
                    if 'tokens' not in dato or 'labels' not in dato:
                        print(f"Error en línea {numero_linea}: Faltan las claves 'tokens' o 'labels'.")
                        print(f"Contenido: {linea_texto.strip()}")
                        errores_encontrados += 1
                        continue

                    tokens = dato['tokens']
                    labels = dato['labels']

                    # Verificar que 'tokens' y 'labels' sean listas
                    if not isinstance(tokens, list) or not isinstance(labels, list):
                        print(f"Error en línea {numero_linea}: 'tokens' o 'labels' no es una lista.")
                        print(f"Tipo de tokens: {type(tokens)}, Tipo de labels: {type(labels)}")
                        print(f"Contenido: {linea_texto.strip()}")
                        errores_encontrados += 1
                        continue

                    # La comprobación principal: comparar longitudes
                    if len(tokens) != len(labels):
                        print(f"¡Desalineación encontrada en la línea {numero_linea}!")
                        print(f"  Número de tokens: {len(tokens)}")
                        print(f"  Número de labels: {len(labels)}")
                        print(f"  Tokens: {tokens[:20]}..." if len(tokens) > 20 else tokens) # Muestra una vista previa
                        # print(f"  Labels: {labels[:20]}..." if len(labels) > 20 else labels) # Descomentar si quieres ver las labels
                        errores_encontrados += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error de formato JSON en la línea {numero_linea}: {e}")
                    print(f"Contenido: {linea_texto.strip()}")
                    errores_encontrados += 1
                except Exception as e:
                    print(f"Error inesperado procesando la línea {numero_linea}: {e}")
                    print(f"Contenido: {linea_texto.strip()}")
                    errores_encontrados += 1
                
                if errores_encontrados > 0 and errores_encontrados % 5 == 0: # Imprime un separador cada 5 errores
                    print("-" * 30)

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_archivo}' no fue encontrado.")
        return

    print("\n--- Verificación Completada ---")
    if lineas_procesadas == 0:
        print("El archivo está vacío o no se pudo leer.")
    elif errores_encontrados == 0:
        print(f"¡Todo perfecto! Las {lineas_procesadas} líneas procesadas en '{ruta_archivo}' tienen tokens y labels alineados.")
    else:
        print(f"Se encontraron {errores_encontrados} errores en {lineas_procesadas} líneas procesadas en '{ruta_archivo}'.")
        print("Por favor, revisa los mensajes anteriores para más detalles.")

if __name__ == "__main__":
    nombre_archivo = "temporal_expressions_dataset_clean.jsonl"
    
    # --- Bloque para crear un archivo de prueba (opcional) ---
    # Si quieres probar el script rápidamente, puedes descomentar este bloque
    # para crear un archivo de ejemplo. Asegúrate de que no sobreescriba tu archivo real.
    # print("Creando archivo de prueba temporal 'temporal_expression_negative_sampling.jsonl'...")
    # test_data = [
    #     {"tokens": ["El", "festival", "es", "en", "Mayo", "."], "labels": [0, 0, 0, 0, 1, 0]}, # Correcto
    #     {"tokens": ["Mayo", "es", "un", "mes", "."], "labels": [1, 0, 0, 0]}, # Incorrecto - Desalineado
    #     {"tokens": ["¿", "Puedo", "irme", "ya", "?"], "labels": [0, 0, 0, 0, 0]}, # Correcto
    #     '{"tokens": ["Otro", "ejemplo"], "labels": [0,0,0,0]}', # JSON válido, pero desalineado
    #     'Esto no es un JSON', # Error de formato JSON
    #     {"text": ["No", "tiene", "tokens"], "info": [0,0,0]}, # Faltan claves
    #     {"tokens": "no es lista", "labels": [0,0]} # Tipo incorrecto
    # ]
    # with open(nombre_archivo, 'w', encoding='utf-8') as f_test:
    #     for item in test_data:
    #         if isinstance(item, dict):
    #             f_test.write(json.dumps(item) + '\n')
    #         else:
    #             f_test.write(item + '\n')
    # print("Archivo de prueba creado.")
    # --- Fin del bloque de prueba ---

    verificar_alineacion_tokens_labels(nombre_archivo)