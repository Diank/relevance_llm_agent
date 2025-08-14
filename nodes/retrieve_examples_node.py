import json
from agent.retriever_builder import build_retrievers

with open("labeled_pairs.json", "r", encoding="utf-8") as f:
    labeled_pairs = json.load(f)

retrievers = build_retrievers(labeled_pairs)
per_rubric_retrievers = retrievers["per_rubric_retrievers"]
fallback_retriever = retrievers["fallback_retriever"]

def retrieve_fewshot_examples_node(state: dict) -> dict:
    print("<🔎 РЕТРИВ ПОХОЖИХ ПРИМЕРОВ ИЗ ТОЙ ЖЕ КАТЕГОРИИ ... >")

    query = state["query"]
    rubric_name = state["normalized_main_rubric_name_ru"]

    search_text = f"Запрос: {query}. Категория: {rubric_name}"
    retriever = per_rubric_retrievers.get(rubric_name)

    if retriever:
        nodes = retriever.retrieve(search_text)
        if not nodes:
            print("[INFO] Ничего не найдено по рубрике — fallback ко всем документам")
            nodes = fallback_retriever.retrieve(search_text)
    else:
        print("[INFO] Нет ретривера для рубрики — fallback ко всем документам")
        nodes = fallback_retriever.retrieve(search_text)

    parsed_examples = []
    for node in nodes:
        text = node.text
        assessor_label = node.metadata.get("assessor_label", "NEED_MORE_INFO")

        q, o, r = "unknown", "unknown", rubric_name
        if "Запрос:" in text and "Ответ:" in text:
            try:
                q = text.split("Запрос:")[1].split("Ответ:")[0].strip().rstrip(".")
                o = text.split("Ответ:")[1].split("Категория:")[0].strip().rstrip(".")
                r = text.split("Категория:")[1].strip()
            except:
                pass

        parsed_examples.append({
            "query": q,
            "org_text": o,
            "normalized_main_rubric_name_ru": r,
            "assessor_label": assessor_label
        })

    return {
        **state,
        "retrieved_fewshot_examples": parsed_examples,
        "step_count": state.get("step_count", 0) + 1
    }
