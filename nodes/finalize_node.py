from agent.schema import State

def logger(state: State):
    print(f"\nЗапрос: [{state.get('query')}]")
    print(f"Организация: [{state.get('org_text')}]")
    print(f"Релевантность")
    print(f"--- llm_label: {state.get('llm_label')}")
    print(f"--- assessor_label: {state.get('assessor_label')}")

def finalize_node(state: State):

    if state.get("llm_label") == "NEED_MORE_INFO":
        state['llm_label'] = "IRRELEVANT"
        logger(state)
        return {
            **state,
            "llm_label": "IRRELEVANT",
        }

    logger(state)

    return {**state}
