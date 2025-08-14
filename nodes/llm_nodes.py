from agent.schema import State, Output
from agent.model import llm, llm4o

from nodes.build_prompt_node import build_prompt_node, parse_response

def llm_node_base(state: State, llm_model) -> State:
    state["step_count"] = state.get("step_count", 0) + 1

    enriched_state = {
        **state,
        "retrieved_fewshot_examples": state.get("retrieved_fewshot_examples", []),
        "search_snippet": state.get("search_snippet", ""),
        "search_result_url": state.get("search_result_url", ""),
    }

    prompt = build_prompt_node(enriched_state)

    chain = prompt | llm_model | (lambda response: parse_response(response.content))
    parsed_output = Output(**chain.invoke(enriched_state))

    return {**state, "llm_label": parsed_output.llm_label}


def llm_node(state: State) -> State:
    return llm_node_base(state, llm)


def llm_node_slow(state: State) -> State:
    return llm_node_base(state, llm4o)

    return {
        **state,
        "llm_label": parsed_output.llm_label,
    }

