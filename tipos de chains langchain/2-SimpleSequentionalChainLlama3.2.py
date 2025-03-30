"""
Script para generar un nombre de empresa y su descripción usando LangChain y Ollama con SimpleSequentialChain.

Descripción:
    Este script utiliza LangChain para crear un nombre de empresa basado en un producto y luego generar una descripción
    de 20 palabras para esa empresa. Carga un modelo de lenguaje (ChatOllama con llama3.2:latest), lee un archivo CSV
    (Data.csv) para mostrar datos iniciales, y emplea SimpleSequentialChain para ejecutar dos cadenas secuenciales:
    una para el nombre y otra para la descripción. Los resultados se imprimen con detalles verbose.

Entradas:
    - Data.csv: Archivo CSV con datos de productos (requerido).
    - .env: Archivo con variables de entorno (opcional, no requerido para Ollama).
    - producto: Cadena de texto con el nombre del producto (hardcoded).

Salidas:
    - Impresión del encabezado del DataFrame desde Data.csv.
    - Impresión verbose de cada paso de la cadena (nombre y descripción).
    - Resultado final con la descripción generada.

Dependencias:
    - langchain, langchain-ollama, pandas, python-dotenv
"""

# Importar módulos para manejar el entorno y variables
import os
from dotenv import load_dotenv, find_dotenv

# Cargar variables de entorno desde el archivo .env
# Esto permite acceder a configuraciones opcionales, aunque no es necesario para Ollama
_ = load_dotenv(find_dotenv())

# Definir el modelo de lenguaje a utilizar
# En este caso, usamos llama3.2:latest de Ollama, ejecutado localmente
llm_model = "llama3.2:latest"

# Importar pandas para trabajar con datos CSV
import pandas as pd

# Leer el archivo CSV y mostrar las primeras filas
# Esto carga Data.csv en un DataFrame y muestra su encabezado para verificar los datos
df = pd.read_csv("Data.csv")
print("Primeras filas del DataFrame:")
print(df.head())

# Importar clases necesarias de LangChain y Ollama
from langchain_ollama import ChatOllama  # Modelo de chat de Ollama
from langchain.prompts import ChatPromptTemplate  # Plantilla para prompts
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
)  # Cadenas para procesar prompts

# Inicializar el modelo de lenguaje
# Configuramos ChatOllama con una temperatura de 0.9 para respuestas creativas
llm = ChatOllama(
    temperature=0.9, 
    model=llm_model, 
    base_url="http://localhost:11434"  # URL del servidor local de Ollama
)

# Crear el primer prompt para generar el nombre de la empresa
# Este prompt toma un producto como entrada y pide un nombre
first_prompt = ChatPromptTemplate.from_template(
    "¿Cuál es el mejor nombre para describir una empresa que fabrica {producto}?"
)

# Definir el producto de entrada
# Este es el valor que se usará en los prompts
producto = "Juego de Sábanas Queen Size"

# Crear la primera cadena (LLMChain)
# Esta cadena combina el modelo y el primer prompt para generar el nombre
#chain_one = LLMChain(llm=llm, prompt=first_prompt)
chain_one = first_prompt | llm

# Crear el segundo prompt para generar la descripción
# Este prompt toma el nombre generado y pide una descripción de 20 palabras
second_prompt = ChatPromptTemplate.from_template(
    "Escribe una descripción de 20 palabras para la siguiente empresa: {nombre_de_empresa}"
)

# Crear la segunda cadena (LLMChain)
# Esta cadena usa el modelo y el segundo prompt para generar la descripción
#chain_two = LLMChain(llm=llm, prompt=second_prompt)
chain_two = second_prompt | llm

# Crear la cadena secuencial simple
# SimpleSequentialChain conecta las dos cadenas: el output de chain_one alimenta chain_two

overall_chain = chain_one | chain_two
# Ejecutar la cadena secuencial con el producto
# Esto genera el nombre y luego la descripción, imprimiendo el proceso y el resultado final
#result = overall_simple_chain.invoke(producto)

# Imprimir el resultado final explícitamente
# El resultado es la salida de la segunda cadena (la descripción)
print("\n> Starting chain execution...")
result = overall_chain.invoke({"producto": producto})
print("\nResultado final:")
print(result.content)