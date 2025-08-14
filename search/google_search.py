import requests
from config import GOOGLE_API_KEY, GOOGLE_CX

def google_pse_search(query, max_results=3):
    """Поиск через Google Programmable Search Engine."""
    print("Поиск через Google PSE")
    try:
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={query}"
        )
        response = requests.get(url)
        data = response.json()

        if "items" not in data:
            print("Google не вернул результатов:", data.get("error", "Пусто"))
            return []

        return [
            {
                "title": item.get("title", ""),
                "body": item.get("snippet", ""),
                "href": item.get("link", "")
            }
            for item in data["items"][:max_results]
        ]
    except Exception as e:
        print(f"Ошибка в google_pse_search: {e}")
        return []
