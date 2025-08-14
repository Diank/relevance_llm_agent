from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from .schema import State

from nodes.search_node import actual_search_node
from nodes.filter_reviews_node import filter_reviews_node
from nodes.llm_nodes import llm_node, llm_node_slow
from nodes.retrieve_examples_node import retrieve_fewshot_examples_node
from nodes.finalize_node import finalize_node


# Логика маршрутизации по релевантности:
# 1. filter_reviews → фильтрация отзывов
# 2. llm_agent_fast → легкая модель gpt-4o-mini
# 3. Если NEED_MORE_INFO:
#    - retrieve_examples → поиск примеров
#    - повторная llm_agent_fast
#    - llm_agent_slow (более дорогая/медленная модель gpt-4o)
#    - search → внешний поиск, затем снова llm_agent_slow
# 4. finalize → завершение


def route_by_relevance(state: State):
    step = state.get("step_count", 0)
    label = state.get("llm_label")
    has_search = bool(state.get("search_snippet"))

    if step > 4:
        return "finalize"

    if step == 1 and label == "NEED_MORE_INFO":
        return "retrieve_examples"

    if step == 2 and label == "NEED_MORE_INFO":
        return "llm_agent_fast"

    if step == 3 and label == "NEED_MORE_INFO":
        return "llm_agent_slow"

    if step == 4 and label == "NEED_MORE_INFO" and not has_search:
        return "search"

    return "finalize"


def build_graph():
    builder = StateGraph(State)

    builder.add_node("filter_reviews", RunnableLambda(filter_reviews_node))
    builder.add_node("llm_agent_fast", RunnableLambda(llm_node))
    builder.add_node("llm_agent_slow", RunnableLambda(llm_node_slow))
    builder.add_node("search", RunnableLambda(actual_search_node))
    builder.add_node("retrieve_examples", RunnableLambda(retrieve_fewshot_examples_node))
    builder.add_node("finalize", RunnableLambda(finalize_node))

    builder.set_entry_point("filter_reviews")

    builder.add_edge("filter_reviews", "llm_agent_fast")
    builder.add_conditional_edges("llm_agent_fast", route_by_relevance)
    builder.add_conditional_edges("llm_agent_slow", route_by_relevance)
    builder.add_conditional_edges("retrieve_examples", route_by_relevance)
    builder.add_conditional_edges("search", route_by_relevance)

    builder.set_finish_point("finalize")

    return builder.compile()
