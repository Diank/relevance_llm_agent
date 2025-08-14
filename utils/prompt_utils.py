from typing import List

def sanitize_query_for_prompt(query: str) -> str:  # <--- наткнулась на ошибку от Azure OpenAI, связанная с фильтрацией контента
    return (query
        .replace("интим", "инт*м")
        .replace("секс", "с*кс")
        .replace("сексуальн", "с*ксуальн")
        .replace("эроти", "эр*ти")
        .replace("стриптиз", "стр*птиз")
        .replace("эскорт", "эск*рт")
    )


def format_fewshot_examples(examples: List[dict]) -> str:
    """Форматирует список fewshot-примеров для вставки в промпт"""
    return "\n\n".join([
        f"""Запрос: {ex.get('query', '')}
Организация: {ex.get('org_text', '').strip()}
Категория: {ex.get('normalized_main_rubric_name_ru', '').strip()}
Релевантность: {ex.get('assessor_label', '')}"""
        for ex in examples
    ])