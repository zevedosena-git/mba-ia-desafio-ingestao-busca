import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()


def run_ingest() -> None:
    """
    Carrega o PDF (document.pdf), divide em chunks, gera embeddings com Google
    e persiste no PGVector. Usa envio em lotes com retry em caso de 429 (quota).
    Exige GOOGLE_API_KEY, PGVECTOR_URL e PGVECTOR_COLLECTION.
    """
    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "document.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    docs = PyPDFLoader(str(pdf_path)).load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)
    if not splits:
        return

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION", ""),
        connection=os.getenv("PGVECTOR_URL", ""),
        use_jsonb=True,
    )

    batch_size = 20
    for i in range(0, len(enriched), batch_size):
        batch_docs = enriched[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        success = False
        while not success:
            try:
                print(f"Enviando bloco {i // batch_size + 1} de {(len(enriched) + batch_size - 1) // batch_size}...")
                store.add_documents(documents=batch_docs, ids=batch_ids)
                success = True
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print("Limite de taxa atingido. Aguardando 60 segundos...")
                    time.sleep(60)
                else:
                    raise e
    print("Processamento concluído com sucesso!")


if __name__ == "__main__":
    run_ingest()
