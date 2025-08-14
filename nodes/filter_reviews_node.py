from utils.text_processing import filter_reviews, build_org_text

def filter_reviews_node(state: dict) -> dict:
    reviews = state.get("reviews_summarized", "")
    cleaned = filter_reviews(reviews)

    updated_state = {
        **state,
        "reviews_summarized": cleaned,
    }
    updated_state["org_text"] = build_org_text(updated_state)
    return updated_state
