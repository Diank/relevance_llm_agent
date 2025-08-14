import time
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

LABEL_MAP = {
    'RELEVANT_PLUS': 1,
    'IRRELEVANT': 0
}

def map_labels(series: pd.Series) -> pd.Series:
    """Преобразует текстовые метки в float по словарю LABEL_MAP."""
    return series.replace(LABEL_MAP).astype(float)


def evaluate_agent(graph, eval_data):
    results = []

    for _, row in eval_data.iterrows():
        input_dict = {
            **row.to_dict(),
            "base_fewshot_examples": [],
            "retrieved_fewshot_examples": [],
            "search_snippet": "",
            "search_result_url": "",
        }
        results.append(graph.invoke(input_dict))
        time.sleep(1.1)  # чтобы не спамить API

    df_preds = pd.DataFrame(results)

    # Приводим метки к числам
    df_preds["llm_label"] = map_labels(df_preds["llm_label"])
    df_preds["assessor_label"] = map_labels(df_preds["assessor_label"])

    df_preds.rename(columns={'query': 'Text', 'assessor_label': 'relevance_new'}, inplace=True)

    acc = accuracy_score(df_preds["relevance_new"], df_preds["llm_label"])

    return df_preds, acc
