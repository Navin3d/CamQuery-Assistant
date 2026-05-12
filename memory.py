from agent import collection


def format_search_result(doc) -> str:
    metadata = doc.metadata or {}
    return (
        f"Video: {metadata.get('video')}\n"
        f"Frame: {metadata.get('frame')}\n"
        f"Timestamp: {metadata.get('timestamp')} sec\n\n"
        f"Analysis:\n{doc.page_content}\n"
    )


def search_video_memory(question: str, top_k: int = 5) -> str:
    if not question.strip():
        return "No question provided."

    results = collection.similarity_search(question, k=top_k)

    if not results:
        return "No matching frames found."

    return "\n".join(format_search_result(doc) for doc in results)
