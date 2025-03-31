from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from typing import Dict
from langchain_ollama import ChatOllama  # Modelo de chat de Ollama

# Cargar variables de entorno desde el archivo .env (por ejemplo, OPENAI_API_KEY)
_ = load_dotenv(find_dotenv())

# Inicializar el modelo de lenguaje
llm_model = "llama3.2:latest"
llm = ChatOllama(
    temperature=0.9,
    model=llm_model,
    base_url="http://localhost:11434",  # URL del servidor local de Ollama
)


# Definir plantillas de prompts para cada área de especialización
physics_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un profesor de física muy inteligente.
Eres excelente respondiendo preguntas sobre física de manera concisa y fácil de entender.
Cuando no sabes la respuesta a una pregunta, admites que no la sabes.
    """,
        ),
        ("human", "{input}"),  # Placeholder para la entrada del usuario
    ]
)

math_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un muy buen matemático.
Eres excelente respondiendo preguntas de matemáticas.
Eres tan bueno porque eres capaz de descomponer problemas difíciles en sus partes componentes,
responder las partes componentes y luego juntarlas para responder la pregunta más amplia.
    """,
        ),
        ("human", "{input}"),
    ]
)

history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un muy buen historiador.
Tienes un excelente conocimiento y comprensión de las personas, eventos y contextos de una variedad
de períodos históricos. Tienes la capacidad de pensar, reflexionar, debatir, discutir y evaluar
el pasado. Tienes respeto por la evidencia histórica y la capacidad de utilizarla para apoyar tus
explicaciones y juicios.
    """,
        ),
        ("human", "{input}"),
    ]
)

cs_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un exitoso científico de la computación.
Tienes una pasión por la creatividad, la colaboración, el pensamiento innovador, la confianza,
fuertes capacidades para resolver problemas, comprensión de teorías y algoritmos, y excelentes
habilidades de comunicación. Eres excelente respondiendo preguntas de programación.
Eres tan bueno porque sabes cómo resolver un problema describiendo la solución en pasos imperativos
que una máquina puede interpretar fácilmente y sabes cómo elegir una solución que tenga un buen
equilibrio entre complejidad temporal y complejidad espacial.
    """,
        ),
        ("human", "{input}"),
    ]
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente general útil."),
        ("human", "{input}"),
    ]  # Prompt por defecto para casos no específicos
)

# Crear cadenas para cada área de especialización
physics_chain = physics_prompt | llm  # Cadena para física: prompt + modelo
math_chain = math_prompt | llm  # Cadena para matemáticas: prompt + modelo
history_chain = history_prompt | llm  # Cadena para historia: prompt + modelo
cs_chain = cs_prompt | llm  # Cadena para ciencias de la computación: prompt + modelo
default_chain = default_prompt | llm  # Cadena por defecto: prompt + modelo


# Función de enrutamiento
def route_query(query: str) -> RunnableLambda:
    """
    Enruta una consulta a la cadena de especialización adecuada según palabras clave.

    Args:
        query (str): La consulta ingresada por el usuario.

    Returns:
        RunnableLambda: La cadena correspondiente (física, matemáticas, historia, ciencias de la computación o por defecto).

    Esta función analiza la consulta en minúsculas y busca palabras clave específicas para determinar
    qué cadena de especialización debe manejar la solicitud. Si no encuentra coincidencias, usa la cadena por defecto.
    """
    query_lower = (
        query.lower()
    )  # Convertir la consulta a minúsculas para coincidencias insensibles a mayúsculas
    if any(
        keyword in query_lower
        for keyword in ["física", "radiación", "cuerpo negro", "energía"]
    ):
        return physics_chain  # Enrutar a física si hay palabras clave relacionadas
    elif any(
        keyword in query_lower
        for keyword in ["cuanto", "+", "-", "*", "/", "matemáticas"]
    ):
        return math_chain  # Enrutar a matemáticas si hay operadores o términos relacionados
    elif any(
        keyword in query_lower
        for keyword in ["quién", "historia", "belgrano", "pasado"]
    ):
        return history_chain  # Enrutar a historia si hay términos históricos o preguntas sobre personas
    elif any(
        keyword in query_lower
        for keyword in ["scheduler", "dispatcher", "so", "computación"]
    ):
        return cs_chain  # Enrutar a ciencias de la computación si hay términos técnicos de programación
    else:
        return default_chain  # Usar la cadena por defecto si no hay coincidencias


# Cadena principal usando RunnableLambda
chain = RunnableLambda(lambda x: route_query(x["input"]).invoke(x["input"]))
# La cadena principal toma una entrada, la enruta con route_query y ejecuta la cadena seleccionada

# Probar la cadena con consultas de ejemplo
pregunta1 = chain.invoke(
    {"input": "¿Qué es la radiación del cuerpo negro?"}
)  # Consulta de física
print("Pregunta 1:", pregunta1.content)  # Imprimir respuesta

pregunta2 = chain.invoke({"input": "Cuanto es 2 + 2"})  # Consulta de matemáticas
print("Pregunta 2:", pregunta2.content)  # Imprimir respuesta

pregunta3 = chain.invoke({"input": "Quien es Manuel Belgrano?"})  # Consulta de historia
print("Pregunta 3:", pregunta3.content)  # Imprimir respuesta

pregunta4 = chain.invoke(
    {
        "input": "Cual es el rol del Scheduler y del Dispatcher en un SO"
    }  # Consulta de ciencias de la computación
)
print("Pregunta 4:", pregunta4.content)  # Imprimir respuesta
