from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI  # Modelo de chat de OpenAI
from langchain.prompts import ChatPromptTemplate  # Plantilla para prompts
from langchain.chains import LLMChain

# Cargar variables de entorno desde el archivo .env
# Esto permite acceder a OPENAI_API_KEY de forma segura
_ = load_dotenv(find_dotenv())

# Definir el modelo de lenguaje a utilizar
temp = 0  # Temperatura del modelo para respuestas más determinísticas
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=temp, model=llm_model)

# Definir plantillas de prompts para diferentes áreas de conocimiento
physics_template = """
Eres un profesor de física muy inteligente.
Eres excelente respondiendo preguntas sobre física de manera concisa y fácil de entender.
Cuando no sabes la respuesta a una pregunta, admites que no la sabes.

Aquí hay una pregunta:
{input}"""

math_template = """
Eres un muy buen matemático.
Eres excelente respondiendo preguntas de matemáticas.
Eres tan bueno porque eres capaz de descomponer
problemas difíciles en sus partes componentes,
responder las partes componentes y luego juntarlas
para responder la pregunta más amplia.

Aquí hay una pregunta:
{input}"""

history_template = """
Eres un muy buen historiador.
Tienes un excelente conocimiento y comprensión de las personas,
eventos y contextos de una variedad de períodos históricos.
Tienes la capacidad de pensar, reflexionar, debatir, discutir y
evaluar el pasado. Tienes respeto por la evidencia histórica
y la capacidad de utilizarla para apoyar tus explicaciones
y juicios.

Aquí hay una pregunta:
{input}"""

computerscience_template = """
Eres un exitoso científico de la computación.
Tienes una pasión por la creatividad, la colaboración,
el pensamiento innovador, la confianza, fuertes capacidades para resolver problemas,
comprensión de teorías y algoritmos, y excelentes habilidades de comunicación.
Eres excelente respondiendo preguntas de programación.

Aquí hay una pregunta:
{input}"""

# Lista de prompts con su respectiva descripción
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "prompt_template": history_template,
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template,
    },
]

# Crear cadenas de destino para cada área del conocimiento
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Crear la lista de destinos en formato de cadena
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Definir la plantilla del enrutador para seleccionar el prompt adecuado
MULTI_PROMPT_ROUTER_TEMPLATE = """Dado un texto de entrada sin procesar para un
modelo de lenguaje, selecciona el prompt del modelo que mejor se ajuste a la entrada.
Se te proporcionarán los nombres de los prompts disponibles y una
descripción de para qué está mejor adaptado cada prompt.

<< FORMATTING >>
Return a markdown code snippet con un objeto JSON con el siguiente formato:
```json
{{{{
    "destination": string  nombre del prompt a usar o "DEFAULT"
    "next_inputs": string  una versión potencialmente modificada del input original
}}}}
```

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
```json"""

# Crear el enrutador de prompts
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Configurar la cadena MultiPromptChain para manejar múltiples áreas de conocimiento
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template("{input}")),
    verbose=True,
)

# Invocar la cadena con distintas preguntas y mostrar las respuestas
pregunta1 = chain.invoke("¿Qué es la radiación del cuerpo negro?")
print(pregunta1["text"])

pregunta2 = chain.invoke("Cuánto es 2 + 2")
print(pregunta2["text"])

pregunta3 = chain.invoke("¿Quién es Manuel Belgrano?")
print(pregunta3["text"])

pregunta4 = chain.invoke("¿Cuál es el rol del Scheduler y del Dispatcher en un SO?")
print(pregunta4["text"])
