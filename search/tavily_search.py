import requests
from config import TAVILY_API_KEY

def tavily_search(query, max_results=3):
    """Fallback-поиск через Tavily."""
    print("Fallback через Tavily")
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results
    }

    r = requests.post(url, headers=headers, json=json_data)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    return [
        {
            "title": item["title"],
            "body": item.get("content", ""),
            "href": item["url"]
        }
        for item in results
    ]
