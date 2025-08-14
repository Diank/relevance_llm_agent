import json
import pandas as pd
from tqdm import tqdm
from utils.text_processing import build_org_text, filter_reviews


def data_preparation_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Очистка и подготовка данных."""
    data = data.copy()

    data["prices_summarized"] = data["prices_summarized"].fillna("Информация отсутствует")
    data["reviews_summarized"] = data["reviews_summarized"].fillna("Информация отсутствует")

    data["assessor_label"] = data["assessor_label"].replace({
        1: 'RELEVANT_PLUS',
        0: 'IRRELEVANT'
    }).astype(str)

    return data


def prepare_labeled_pairs(train_data: pd.DataFrame, sample_size: int = 10) -> list:
    """Формирует список пар query/org_text для обучения ретривера."""
    labeled_pairs = []

    for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
        clean_reviews = filter_reviews(row["reviews_summarized"])

        org_text_clean = build_org_text((
            row.get("name", ""),
            row.get("address", ""),
            row.get("prices_summarized", ""),
            clean_reviews
        ))

        labeled_pairs.append({
            "query": row["query"],
            "org_text": org_text_clean,
            "normalized_main_rubric_name_ru": row["normalized_main_rubric_name_ru"],
            "assessor_label": row["assessor_label"]
        })

    return labeled_pairs


def save_labeled_pairs(labeled_pairs: list, filename: str = "labeled_pairs.json"):
    """Сохраняет пары в JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(labeled_pairs, f, ensure_ascii=False, indent=2)
