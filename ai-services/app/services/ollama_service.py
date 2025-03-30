import requests
from app.config import settings
from app.models.request_model import QueryRequest
from app.models.response_model import QueryResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings


class OllamaService:
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        # other params...
    )
    vectorstore = DocArrayInMemorySearch.from_texts(
        [
            "El sol es una estrella.",
            "Los delfines son mamíferos.",
            "La Torre Eiffel está en París.",
            "Las ballenas azules son los animales más grandes del planeta.",
            "El agua cubre el 70 porciento de la superficie de la Tierra.",
            "El ajedrez es un juego de estrategia muy antiguo.",
            "Las medusas existen desde hace más de 500 millones de años.",
            "El español es el segundo idioma más hablado en el mundo.",
            "Las hormigas pueden cargar hasta 50 veces su propio peso.",
            "El café se originó en Etiopía.",
            "El Sahara es el desierto más grande del mundo.",
            "El cerebro humano tiene alrededor de 86 mil millones de neuronas.",
            "La Gran Muralla China mide más de 21,000 kilómetros.",
            "Los gatos tienen 32 músculos en cada oreja.",
            "Los koalas duermen hasta 22 horas al día.",
        ],
        embedding=OllamaEmbeddings(
            model="llama3.2:latest",
        ),
    )
    retriever = vectorstore.as_retriever()

    def simple_query_api(query: QueryRequest) -> QueryResponse:
        url = f"{settings.OLLAMA_API_URL}/api/generate"
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url,
            json={"model": "llama3.2:latest", "prompt": query.prompt, "stream": False},
            headers=headers,
        )

        if response.status_code == 200:
            return QueryResponse(result=response.json().get("response", ""))
        else:
            response.raise_for_status()

    def chat_api(query: QueryRequest) -> QueryResponse:
        url = f"{settings.OLLAMA_API_URL}/api/chat"
        headers = {"Content-Type": "application/json"}
        messages = [
            {"role": "assistant", "content": "You are a senior Java programmer."},
            {"role": "user", "content": query.prompt},
        ]
        response = requests.post(
            url,
            json={"model": "llama3.2:latest", "messages": messages, "stream": False},
            headers=headers,
        )

        if response.status_code == 200:
            return QueryResponse(result=response.json()["message"]["content"])
        else:
            response.raise_for_status()

    @classmethod
    def chat_langchain(cls, query: QueryRequest) -> QueryResponse:
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", query.prompt),
        ]

        ai_msg = cls.llm.invoke(messages)

        return QueryResponse(result=ai_msg.content)

    @classmethod
    def chat_with_template(cls, query: QueryRequest) -> QueryResponse:
        template = """Responda la pregunta basándose únicamente en el siguiente contexto:
            {context}

            Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        chain = (
            RunnableMap(
                {
                    "context": lambda x: cls.retriever.invoke(x["question"]),
                    "question": lambda x: x["question"],
                }
            )
            | prompt
            | cls.llm
            | output_parser
        )

        ai_msg = chain.invoke({"question": query.prompt})

        return QueryResponse(result=ai_msg)
