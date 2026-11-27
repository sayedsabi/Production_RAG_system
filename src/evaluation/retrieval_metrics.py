
def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(top_k) if top_k else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0
