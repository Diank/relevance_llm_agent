from sklearn.metrics.pairwise import cosine_similarity
from agent.model import embed_model

def get_embedding(text: str):
    """Возвращает эмбеддинг строки."""
    if not text.strip():
        return None
    return embed_model.embed_query(text)

def select_best_result(results: list, reference_text: str):
    """Выбирает результат с наибольшим сходством с reference_text."""
    if not results:
        return "Ничего не найдено", None

    ref_emb = get_embedding(reference_text)
    if not ref_emb:
        return "Ошибка при создании эмбеддинга", None

    best_score = -1
    best_result = None

    for r in results:
        combined = f"{r.get('title', '')} {r.get('body', '')}"
        result_emb = get_embedding(combined)
        if not result_emb:
            continue
        score = cosine_similarity([ref_emb], [result_emb])[0][0]
        print(f"Similarity: {score:.4f} | {combined[:60]}...")
        if score > best_score:
            best_score = score
            best_result = r

    if not best_result:
        return "Ничего не найдено", None

    summary = (
        f"Title: {best_result['title']}\n"
        f"Snippet: {best_result['body']}\n"
        f"URL: {best_result['href']}"
    )
    return summary, best_result["href"]
