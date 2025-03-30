"""
Script para generar un nombre de empresa basado en un producto usando LangChain y Ollama.

Descripción / Description:
    Este script carga un modelo de lenguaje (Ollama con llama3.2:latest) y utiliza una plantilla
    de prompt para generar un nombre de empresa basado en un producto dado. Lee un archivo CSV
    (Data.csv) para mostrar datos iniciales, configura el entorno desde un archivo .env, y emplea
    la sintaxis moderna de LangChain (prompt | llm) para procesar el input y obtener una respuesta
    del modelo. Imprime tanto el prompt enviado como la respuesta generada.

Entradas / Inputs:
    - Data.csv: Un archivo CSV con datos de productos (requerido).
    - .env: Un archivo con variables de entorno (opcional, no requerido para Ollama si se usa el default).
    - producto: Una cadena de texto que describe el producto (hardcoded en este caso).

Salidas / Outputs:
    - Impresión del encabezado del DataFrame desde Data.csv.
    - Impresión del prompt formateado enviado al modelo.
    - Impresión del nombre de empresa sugerido por el modelo.

Dependencias / Dependencies:
    - langchain, langchain-ollama, pandas, python-dotenv
"""

import os
from dotenv import load_dotenv, find_dotenv

# Cargar variables de entorno / Load environment variables
_ = load_dotenv(find_dotenv())

# Definir el modelo / Define the model
llm_model = "llama3.2:latest"

import pandas as pd

# Leer y mostrar el CSV / Read and display the CSV
df = pd.read_csv("Data.csv")
print("Primeras filas del DataFrame / First rows of DataFrame:")
print(df.head())

from langchain_ollama import ChatOllama  # Usar ChatOllama en lugar de Ollama
from langchain.prompts import ChatPromptTemplate

# Inicializar el modelo / Initialize the model
llm = ChatOllama(temperature=0.1, model=llm_model, base_url="http://localhost:11434")

# Crear la plantilla de prompt / Create the prompt template
prompt = ChatPromptTemplate.from_template(
    "¿Cuál es el mejor nombre para describir \
    una empresa que fabrica {producto}?"
)

# Crear la cadena / Create the chain
chain = prompt | llm

# Invocar la cadena con el producto / Invoke the chain with the product
producto = "Juego de Sábanas Queen Size"
result = chain.invoke({"producto": producto})

# Imprimir resultados / Print results
print("\nContenido del mensaje recibido / Received message content:")
print(prompt.format(producto=producto))  # Mostrar el prompt / Show the prompt
print("\nRespuesta del modelo / Model response:")
print(result.content)  # ChatOllama devuelve un objeto con .content
