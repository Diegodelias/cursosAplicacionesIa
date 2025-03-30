"""
Script para generar un nombre de empresa y su descripción usando LangChain y OpenAI con SimpleSequentialChain.

Descripción:
    Este script utiliza LangChain para crear un nombre de empresa basado en un producto y luego generar una descripción
    de 20 palabras para esa empresa. Carga un modelo de lenguaje (ChatOpenAI con gpt-3.5-turbo), lee un archivo CSV
    (Data.csv) para mostrar datos iniciales, y emplea SimpleSequentialChain para ejecutar dos cadenas secuenciales:
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
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
)  # Cadenas para procesar prompts

# Inicializar el modelo de lenguaje
# Configuramos ChatOpenAI con una temperatura de 0.9 para respuestas creativas
llm = ChatOpenAI(
    temperature=0.9, model=llm_model, openai_api_key=os.getenv("OPENAI_API_KEY")
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
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Crear el segundo prompt para generar la descripción
# Este prompt toma el nombre generado y pide una descripción de 20 palabras
second_prompt = ChatPromptTemplate.from_template(
    "Escribe una descripción de 20 palabras para la siguiente empresa: {nombre_de_empresa}"
)

# Crear la segunda cadena (LLMChain)
# Esta cadena usa el modelo y el segundo prompt para generar la descripción
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Crear la cadena secuencial simple
# SimpleSequentialChain conecta las dos cadenas: el output de chain_one alimenta chain_two
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],  # Lista de cadenas a ejecutar en orden
    verbose=True,  # Muestra detalles de cada paso durante la ejecución
)

# Ejecutar la cadena secuencial con el producto
# Esto genera el nombre y luego la descripción, imprimiendo el proceso y el resultado final
result = overall_simple_chain.invoke(producto)

# Imprimir el resultado final explícitamente
# El resultado es la salida de la segunda cadena (la descripción)
print("\nResultado final:")
print(result["output"])


# Comienzo ejemplo SequentialChain(lo otro venia de los ejemplos anteriores)
print("Comienzo ejecución código ejemplo SequentialChain")

from langchain.chains import (
    SequentialChain,
)  # Cadena secuencial para múltiples entradas/salidas

# Reutilizar el modelo de lenguaje ya inicializado
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# Plantilla de prompt 1: traducir una reseña al inglés
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:" "\n\n{Review}"
)

# Cadena 1: traduce la reseña al inglés
"""Cadena que toma una reseña como entrada y genera su traducción al inglés."""
chain_one = LLMChain(
    llm=llm, prompt=first_prompt, output_key="English_Review"
)  # Nota: LLMChain está deprecado va a tirar mensaje relacionado con eso pero va a funcionar igual

# Plantilla de prompt 2: resumir la reseña traducida
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:" "\n\n{English_Review}"
)

# Cadena 2: resume la reseña traducida
"""Cadena que toma la reseña traducida y genera un resumen en una oración."""
chain_two = LLMChain(
    llm=llm, prompt=second_prompt, output_key="summary"
)  # Nota: LLMChain está deprecado

# Plantilla de prompt 3: detectar el idioma de la reseña
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)

# Cadena 3: identifica el idioma de la reseña
"""Cadena que analiza la reseña original y determina su idioma."""
chain_three = LLMChain(
    llm=llm, prompt=third_prompt, output_key="language"
)  # Nota: LLMChain está deprecado

# Plantilla de prompt 4: generar un mensaje de seguimiento
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)

# Cadena 4: genera un mensaje de seguimiento
"""Cadena que genera una respuesta de seguimiento basada en el resumen y el idioma detectado."""
chain_four = LLMChain(
    llm=llm, prompt=fourth_prompt, output_key="followup_message"
)  # Nota: LLMChain está deprecado

# Cadena secuencial completa
"""
Cadena secuencial que procesa una reseña a través de cuatro pasos:
1. Traduce la reseña al inglés.
2. Resume la reseña.
3. Detecta el idioma original.
4. Genera un mensaje de seguimiento.
"""
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],  # Cadenas a ejecutar
    input_variables=["Review"],  # Variable de entrada
    output_variables=[
        "English_Review",
        "summary",
        "followup_message",
    ],  # Variables de salida
    verbose=True,  # Muestra detalles del proceso
)

# Obtener la reseña del DataFrame y ejecutar la cadena
review = df.Review[7]  # Selecciona la reseña en la posición 5 del DataFrame
resultado = overall_chain.invoke(review)  # Ejecuta la cadena y almacena el resultado


try:
    print("Reseña traducida al inglés:", resultado["English_Review"])
    print("Resumen:", resultado["summary"])
    print("Mensaje de seguimiento:", resultado["followup_message"])
except KeyError as e:
    print(f"Error: No se encontró la clave {e} en el resultado")
except TypeError as e:
    print(f"Error: resultado no es un diccionario, tipo encontrado: {type(resultado)}")
