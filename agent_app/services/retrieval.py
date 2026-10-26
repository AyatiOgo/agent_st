
from functools import lru_cache
from sentence_transformers import CrossEncoder
from agent_app.services.query_expansion import expand_query
from agent_app.services.vector_store import get_handbook_collection

MIN_RERANK_SCORE = 0.0

# Load the reranker only when it is first needed.
@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def search_handbook(question, handbook_id=None, top_k=5, candidate_k=15):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or candidate_k < 1:
        raise ValueError("top_k and candidate_k must be at least 1.")

    collection = get_handbook_collection()

    if collection.count() == 0:
        return []

    # 1. Filter results to the selected handbook
    where = {"handbook_id": handbook_id} if handbook_id is not None else None

    if where:
        matching = collection.get(where=where, include=["metadatas"])
        available_count = len(matching["ids"])
    else:
        available_count = collection.count()

    if available_count == 0:
        return []

    # 2. Generate the original question and alternative queries
    queries = expand_query(question)

    # 3. Retrieve candidates for every query
    candidates = {}

    for query in queries:
        query_args = {
            "query_texts": [query],
            "n_results": min(candidate_k, available_count),
            "include": ["documents", "metadatas", "distances"],
        }

        if where is not None:
            query_args["where"] = where

        retrieved = collection.query(**query_args)

        for chunk_id, content, metadata, distance in zip(
            retrieved["ids"][0],
            retrieved["documents"][0],
            retrieved["metadatas"][0],
            retrieved["distances"][0],
        ):
            # Keep each chunk only once
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "handbook_id": metadata["handbook_id"],
                    "content": content,
                    "page_start": metadata["page"],
                    "page_end": metadata["page"],
                    "section": "",
                    "distance": float(distance),
                }

    if not candidates:
        return []

    # 4. Rerank all unique candidates against the ORIGINAL question
    candidate_list = list(candidates.values())
    reranker = get_reranker()

    pairs = [
        (question, candidate["content"])
        for candidate in candidate_list
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidate_list, scores):
        candidate["rerank_score"] = float(score)

    # 5. Rank by cross-encoder score, highest first
    candidate_list.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidate_list[:top_k]

