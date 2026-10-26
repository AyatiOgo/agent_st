
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agent_prod.settings")
django.setup()

from agent_app.models import HandbookModel
from agent_app.services.query_expansion import expand_query
from agent_app.services.vector_store import get_handbook_collection
from agent_app.services.retrieval import get_reranker


def test_expanded_retrieval():
    question = "What support services are available to students?"

    handbook = HandbookModel.objects.filter(
        status="completed"
    ).first()

    if handbook is None:
        print("No completed handbook found.")
        return

    collection = get_handbook_collection()
    queries = expand_query(question)

    print(f"Original question: {question}")
    print(f"Handbook ID: {handbook.id}")
    print(f"\nGenerated queries: {len(queries)}")

    # 1. Retrieve and deduplicate chunks from all expanded queries
    candidates = {}

    for query in queries:
        print(f"\nQUERY: {query}")

        retrieved = collection.query(
            query_texts=[query],
            n_results=min(20, collection.count()),
            where={"handbook_id": handbook.id},
            include=["documents", "metadatas", "distances"],
        )

        for chunk_id, content, metadata, distance in zip(
            retrieved["ids"][0],
            retrieved["documents"][0],
            retrieved["metadatas"][0],
            retrieved["distances"][0],
        ):
            print(
                f"Page: {metadata['page']} | "
                f"Distance: {distance:.4f}"
            )
            print(content[:200])
            print("-" * 50)

            # Keep each unique chunk only once
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": content,
                    "page": metadata["page"],
                    "distance": float(distance),
                }

    if not candidates:
        print("\nNo chunks retrieved.")
        return

    # 2. Rerank all candidates against the ORIGINAL question
    reranker = get_reranker()

    candidate_list = list(candidates.values())

    pairs = [
        (question, candidate["content"])
        for candidate in candidate_list
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidate_list, scores):
        candidate["rerank_score"] = float(score)

    # 3. Sort by cross-encoder score, highest first
    ranked_results = sorted(
        candidate_list,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("FINAL RERANKED RESULTS")
    print("=" * 70)

    for result in ranked_results[:10]:
        print(
            f"\nPage: {result['page']} | "
            f"Rerank score: {result['rerank_score']:.4f} | "
            f"Distance: {result['distance']:.4f}"
        )
        print(result["content"][:300])
        print("-" * 70)


if __name__ == "__main__":
    test_expanded_retrieval()
