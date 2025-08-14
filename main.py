from pathlib import Path

# --- Импорты функций ---
from agent.data_loader import load_data
from agent.pipeline import data_preparation_pipeline, prepare_labeled_pairs, save_labeled_pairs
from agent.graph import build_graph
from agent.evaluator import evaluate_agent


if __name__ == "__main__":
    # === 1. Загружаем данные ===
    train_data, eval_data = load_data()
    print(f"Train size: {len(train_data)} | Eval size: {len(eval_data)}")

    # === 2. Подготавливаем тренировочные данные ===
    train_data = data_preparation_pipeline(train_data)
    eval_data = data_preparation_pipeline(eval_data)

    # === 3. Формируем пары для ретривера и сохраняем ===
    labeled_pairs = prepare_labeled_pairs(train_data)
    save_labeled_pairs(labeled_pairs)
    print(f"✅ Сохранено {len(labeled_pairs)} пар в labeled_pairs.json")

    # построение
    app = build_graph()
    print("✅ Граф построен")

    # пример прогона запроса
    example_state = {
        "query": "пицца рядом",
        "name": "Пиццерия Рома",
        "address": "ул. Ленина, 5",
        "prices_summarized": "Средний чек 500р",
        "reviews_summarized": "Очень вкусно",
        "normalized_main_rubric_name_ru": "Пиццерии",
        "assessor_label": "RELEVANT_PLUS"
    }
    example_result = app.invoke(example_state)
    print("\n----> Пример результата для тестового запроса:")
    print(example_result)

    print("\n----> Запуск оценки...")
    df_preds, acc = evaluate_agent(app, eval_data)

    output_path = Path("eval_predictions.json")
    df_preds.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"\n----> Предсказания сохранены в {output_path}")

    print(f"\nAccuracy: {acc:.4f}")
