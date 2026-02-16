import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_postgres import PGVector

from search import PROMPT_TEMPLATE, CONTEXT_SEPARATOR
from ingest import run_ingest

load_dotenv()

LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
K_RESULTS = 10


def _get_store() -> PGVector:
    """
    Cria e retorna o store PGVector (embeddings Google + conexão/coleção do .env).
    Usado para verificar se há dados e para buscar trechos relevantes por pergunta.
    """
    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION", ""),
        connection=os.getenv("PGVECTOR_URL", ""),
        use_jsonb=True,
    )


def _pgvector_has_data() -> bool:
    """
    Verifica se o banco vetorial existe e contém pelo menos um documento.
    Retorna False em caso de erro (ex.: tabela inexistente) ou coleção vazia.
    """
    try:
        store = _get_store()
        docs = store.similarity_search(" ", k=1)
        return len(docs) > 0
    except Exception:
        return False


def get_relevant_context(question: str, k: int = 10) -> str:
    """
    Vetoriza a pergunta, busca os k trechos mais similares no PGVector e retorna
    o conteúdo concatenado com CONTEXT_SEPARATOR para uso no prompt da LLM.
    """
    store = _get_store()
    docs = store.similarity_search_with_score(question, k=k)
    if not docs:
        return ""
    return CONTEXT_SEPARATOR.join(doc.page_content for doc, _ in docs)


def main() -> None:
    """
    Ponto de entrada do chat em linha de comando: valida env, executa ingest se
    o banco estiver vazio, inicia o loop de perguntas e respostas via LLM Gemini.
    """
    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            print(f"Erro: variável de ambiente {k} não definida.")
            return

    if not _pgvector_has_data():
        print("Banco vetorial vazio ou inexistente. Executando ingestão...")
        run_ingest()
        print()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    print("Chat (baseado no documento). Digite 'sair' para encerrar.\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo.")
            break
        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Até logo.")
            break

        contexto = get_relevant_context(pergunta, k=K_RESULTS)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
        msg = llm.invoke([HumanMessage(content=prompt)])
        print(f"\nAssistente: {msg.content}\n")


if __name__ == "__main__":
    main()
