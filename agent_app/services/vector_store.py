
import chromadb

from django.conf import settings
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


CHROMA_PATH = settings.BASE_DIR / "chroma_db"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


def get_handbook_collection():
    return client.get_or_create_collection(
        name="admissions_handbook",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(handbook_id, chunks):
    collection = get_handbook_collection()

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"handbook_{handbook_id}_chunk_{index}")
        documents.append(chunk.page_content)

        metadatas.append({
            "handbook_id": handbook_id,
            "chunk_index": index,
            "page": chunk.metadata["page"],
            "source": str(chunk.metadata["source"]),
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return len(ids)
