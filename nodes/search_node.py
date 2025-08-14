from search.google_search import google_pse_search
from search.tavily_search import tavily_search
from search.utils import select_best_result

def actual_search_node(state: dict) -> dict:
    """Нода для поиска информации по запросу."""
    query = state.get("query", "")
    reference = f"{state.get('name', '')} {state.get('address', '')}".strip()

    try:
        print(f"Запрос: {query}")
        try:
            results = google_pse_search(query)
        except Exception as e:
            print(f"Google PSE ошибка: {e}")
            results = tavily_search(query)

        summary, link = select_best_result(results, reference)

    except Exception as e:
        print("Ошибка при поиске:", e)
        summary, link = "Ничего не найдено", None

    return {
        **state,
        "search_snippet": summary,
        "search_result_url": link,
        "step_count": state.get("step_count", 0) + 1
    }
