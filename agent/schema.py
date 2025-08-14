from typing import Optional, List, TypedDict, Literal
from pydantic import BaseModel

class FewshotExample(TypedDict):
    query: str
    org_text: str
    assessor_label: str

class State(TypedDict):
    query: str
    address: str
    name: str
    normalized_main_rubric_name_ru: str
    prices_summarized: str
    reviews_summarized: str
    permalink: Optional[int]
    org_text: str
    assessor_label: str
    base_fewshot_examples: Optional[List[FewshotExample]]
    step_count: int
    retrieved_fewshot_examples: Optional[List[FewshotExample]]
    search_snippet: Optional[str]
    search_result_url: Optional[str]
    llm_label: Optional[str]

class Output(BaseModel):
    llm_label: Literal["RELEVANT_PLUS", "IRRELEVANT", "NEED_MORE_INFO"]
