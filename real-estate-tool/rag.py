import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

_llm = None
_embedding_function = None


def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_function


def get_llm():
    global _llm
    if _llm is None:
        model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        _llm = ChatGroq(model=model_name, temperature=0.2, max_tokens=700)
    return _llm


def process_urls(urls: list[str], progress_callback=None):
    """
    Load public URLs, split their text, and create a fresh in-memory Chroma vector store.
    This function does not write a shared vector DB to disk, so it is safer for Streamlit Cloud.
    """
    if not urls:
        raise ValueError("At least one URL is required.")

    def update(message: str):
        if progress_callback:
            progress_callback(message)

    update("Initializing embedding model...")
    embedding_function = get_embedding_function()

    update("Loading data from URLs...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    loaded_documents = loader.load()

    loaded_documents = [doc for doc in loaded_documents if doc.page_content and doc.page_content.strip()]
    if not loaded_documents:
        raise RuntimeError("No readable text was found in the provided URLs.")

    update("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(loaded_documents)
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if not docs:
        raise RuntimeError("The URLs loaded, but no useful text chunks were created.")

    update("Creating session vector store...")
    vector_store = Chroma(
        collection_name=f"real_estate_{uuid4().hex}",
        embedding_function=embedding_function,
    )

    update("Adding chunks to vector store...")
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    update("URLs processed successfully.")
    return vector_store, len(docs)


def generate_answer(query: str, vector_store):
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized. Process URLs first.")

    docs = vector_store.similarity_search(query, k=4)
    if not docs:
        return "I could not find relevant information in the processed URLs.", []

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )

    prompt = f"""
You are a helpful research assistant. Answer the question using only the context below.
If the context does not contain the answer, say that the processed URLs do not provide enough information.
Keep the answer clear and concise. Mention source URLs when useful.

Context:
{context}

Question: {query}
"""

    response = get_llm().invoke(prompt)
    answer = response.content
    sources = sorted({doc.metadata.get("source", "Unknown") for doc in docs})
    return answer, sources
