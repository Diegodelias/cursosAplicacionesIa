"""
Script para generar un nombre de empresa basado en un producto usando LangChain y OpenAI.

Descripción / Description:
    Este script carga un modelo de lenguaje (ChatOpenAI con gpt-3.5-turbo) y utiliza una plantilla
    de prompt para generar un nombre de empresa basado en un producto dado. Lee un archivo CSV
    (Data.csv) para mostrar datos iniciales, configura el entorno desde un archivo .env, y emplea
    la sintaxis moderna de LangChain (prompt | llm) para procesar el input y obtener una respuesta
    del modelo. Imprime tanto el prompt enviado como la respuesta generada.

Entradas / Inputs:
    - Data.csv: Un archivo CSV con datos de productos (requerido).
    - .env: Un archivo con la variable OPENAI_API_KEY (requerido).
    - producto: Una cadena de texto que describe el producto (hardcoded en este caso).

Salidas / Outputs:
    - Impresión del encabezado del DataFrame desde Data.csv.
    - Impresión del prompt formateado enviado al modelo.
    - Impresión del nombre de empresa sugerido por el modelo.

Dependencias / Dependencies:
    - langchain, langchain-openai, openai, pandas, python-dotenv
"""

import os
from dotenv import load_dotenv, find_dotenv
import langchain
import langchain_openai
import langchain_ollama



_ = load_dotenv(find_dotenv())

llm_model = "gpt-3.5-turbo"

import pandas as pd

df = pd.read_csv("Data.csv")
df.head()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)

prompt = ChatPromptTemplate.from_template(
    "¿Cuál es el mejor nombre para describir \
    una empresa que fabrica {producto}?"
)

chain = prompt | llm

producto = "Juego de Sábanas Queen Size"
result = chain.invoke(producto)

print("\nContenido del mensaje recibido / Received message content:")
print(result.content)
print("\nRespuesta del modelo / Model response:")
