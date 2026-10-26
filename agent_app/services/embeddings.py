
from django.db import transaction
from agent_app.models import DocumentChunkModel
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once when this module is imported.
model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text):
    """
    Convert text into a 384-dimensional vector.
    """
    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector.tolist()

def embed_handbook_chunks(handbook_id, batch_size=32):
    """
    Generate embeddings for chunks belonging
    to a particular handbook.
    """
    chunks = DocumentChunkModel.objects.filter(
        handbook_id=handbook_id,
        embedding=[],
    ).order_by("chunk_index")

    total_embedded = 0

    while True:
        batch = list(chunks[:batch_size])

        if not batch:
            break

        vectors = model.encode(
            [chunk.content for chunk in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        with transaction.atomic():
            for chunk, vector in zip(batch, vectors):
                chunk.embedding = vector.tolist()

            DocumentChunkModel.objects.bulk_update(
                batch,
                ["embedding"],
            )

        total_embedded += len(batch)

    return {
        "handbook_id": handbook_id,
        "embedded_chunks": total_embedded,
    }