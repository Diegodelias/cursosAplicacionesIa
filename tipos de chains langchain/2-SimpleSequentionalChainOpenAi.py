"""
Script para generar un nombre de empresa y su descripción usando LangChain y OpenAI con RunnableSequence.

Descripción:
    Este script utiliza LangChain para crear un nombre de empresa basado en un producto y luego generar una descripción
    de 20 palabras para esa empresa. Carga un modelo de lenguaje (ChatOpenAI con gpt-3.5-turbo), lee un archivo CSV
    (Data.csv) para mostrar datos iniciales, y emplea RunnableSequence para ejecutar dos cadenas secuenciales:
    una para el nombre y otra para la descripción. Los resultados se imprimen con detalles verbose.

Entradas:
    - Data.csv: Archivo CSV con datos de productos (requerido).
    - .env: Archivo con la variable OPENAI_API_KEY (requerido).
    - producto: Cadena de texto con el nombre del producto (hardcoded).

Salidas:
    - Impresión del encabezado del DataFrame desde Data.csv.
    - Impresión verbose de cada paso de la cadena (nombre y descripción).
    - Resultado final con la descripción generada.

Dependencias:
    - langchain, langchain-openai, openai, pandas, python-dotenv
"""

# Importar módulos para manejar el entorno y variables
import os
from dotenv import load_dotenv, find_dotenv

# Cargar variables de entorno desde el archivo .env
# Esto permite acceder a OPENAI_API_KEY de forma segura
_ = load_dotenv(find_dotenv())

# Definir el modelo de lenguaje a utilizar
# En este caso, usamos gpt-3.5-turbo de OpenAI
llm_model = "gpt-3.5-turbo"

# Importar pandas para trabajar con datos CSV
import pandas as pd

# Leer el archivo CSV y mostrar las primeras filas
# Esto carga Data.csv en un DataFrame y muestra su encabezado para verificar los datos
df = pd.read_csv("Data.csv")
print("Primeras filas del DataFrame:")
print(df.head())

# Importar clases necesarias de LangChain y OpenAI
from langchain_openai import ChatOpenAI  # Modelo de chat de OpenAI
from langchain.prompts import ChatPromptTemplate  # Plantilla para prompts

# Inicializar el modelo de lenguaje
# Configuramos ChatOpenAI con una temperatura de 0.9 para respuestas creativas
llm = ChatOpenAI(
    temperature=0.9, model=llm_model, openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Crear los prompts
first_prompt = ChatPromptTemplate.from_template(
    "¿Cuál es el mejor nombre para describir una empresa que fabrica {producto}?"
)

second_prompt = ChatPromptTemplate.from_template(
    "Escribe una descripción de 20 palabras para la siguiente empresa: {nombre_de_empresa}"
)

# Definir el producto de entrada
# Este es el valor que se usará en los prompts
producto = "Juego de Sábanas Queen Size"

# Crear las cadenas usando el nuevo enfoque con RunnableSequence
chain_one = first_prompt | llm
chain_two = second_prompt | llm

# Crear la cadena secuencial
overall_chain = chain_one | chain_two

# Ejecutar la cadena secuencial con el producto
print("\n> Starting chain execution...")
result = overall_chain.invoke({"producto": producto})
print("\nResultado final:")
print(result.content)
