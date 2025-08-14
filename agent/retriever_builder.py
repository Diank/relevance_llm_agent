from collections import defaultdict
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore


def build_retrievers(labeled_pairs: List[Dict[str, Any]], similarity_top_k: int = 3):
    """
    Создаёт per-rubric и fallback ретриверы на основе размеченных пар.

    Args:
        labeled_pairs: список словарей с ключами
            query, org_text, normalized_main_rubric_name_ru, assessor_label
        similarity_top_k: количество возвращаемых ближайших примеров

    Returns:
        dict: {
            "per_rubric_retrievers": dict[str, Retriever],
            "fallback_retriever": Retriever
        }
    """

    # модель эмбеддингов
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

    # оптимальный сплиттер
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64, include_metadata=True)

    # собираем документы по рубрикам
    documents_by_rubric = defaultdict(list)
    all_documents = []

    for pair in labeled_pairs:
        rubric_name = pair["normalized_main_rubric_name_ru"]
        query = pair["query"]
        org_text = pair["org_text"]

        text = f"Запрос: {query}. Ответ: {org_text}. Категория: {rubric_name}"
        metadata = {
            "assessor_label": pair["assessor_label"]
        }

        doc = Document(text=text, metadata=metadata)
        documents_by_rubric[rubric_name].append(doc)
        all_documents.append(doc)

    # fallback-ретривер (по всем документам)
    fallback_vector_store = SimpleVectorStore()
    fallback_storage_context = StorageContext.from_defaults(vector_store=fallback_vector_store)

    fallback_index = VectorStoreIndex.from_documents(
        documents=all_documents,
        storage_context=fallback_storage_context,
        embed_model=embed_model,
        text_splitter=text_splitter,
    )
    fallback_retriever = fallback_index.as_retriever(similarity_top_k=similarity_top_k)

    # ретриверы по рубрикам
    per_rubric_retrievers = {}
    for rubric, docs in documents_by_rubric.items():
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_context,
            embed_model=embed_model,
            text_splitter=text_splitter,
        )

        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        per_rubric_retrievers[rubric] = retriever

    return {
        "per_rubric_retrievers": per_rubric_retrievers,
        "fallback_retriever": fallback_retriever
    }
