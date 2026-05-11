from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .config import Settings
from .retrieval import HybridRetriever


class RagState(TypedDict):
    query: str
    query_type: str
    docs: List[Document]
    answer: str


def build_graph(settings: Settings):
    retriever = HybridRetriever(settings)
    llm = ChatOpenAI(model=settings.chat_model, temperature=0)

    classifier_prompt = ChatPromptTemplate.from_template(
        """
Classify the user request into exactly one of these labels:
- symbol_lookup (find exact usage, symbols, configs, strings)
- architecture (why/how components exist and connect)
- migration_plan (create phased refactor/migration plans)
- change_history (questions about when/why code changed)

Question: {query}
Return only the label.
""".strip()
    )

    answer_prompt = ChatPromptTemplate.from_template(
        """
You are an internal engineering copilot.
Use only the retrieved context to answer.
If uncertain, state uncertainty clearly.

User question:
{query}

Retrieved context:
{context}

Output requirements:
1) Give a concise answer first.
2) Then include bullet points with evidence.
3) End with citations as file paths or commit SHAs from metadata.
""".strip()
    )

    def classify_query(state: RagState) -> Dict[str, Any]:
        label = (classifier_prompt | llm).invoke({"query": state["query"]}).content.strip().lower()
        allowed = {"symbol_lookup", "architecture", "migration_plan", "change_history"}
        if label not in allowed:
            label = "architecture"
        return {"query_type": label}

    def retrieve_context(state: RagState) -> Dict[str, Any]:
        query = state["query"]
        if state["query_type"] == "symbol_lookup":
            docs = retriever.search(query, k=max(settings.top_k, 10))
        elif state["query_type"] == "migration_plan":
            docs = retriever.search(query + " dependencies interfaces schema usage", k=max(settings.top_k, 10))
        elif state["query_type"] == "change_history":
            docs = retriever.search(query + " commit history why changed", k=max(settings.top_k, 10))
        else:
            docs = retriever.search(query + " architecture service boundaries flow", k=settings.top_k)
        return {"docs": docs}

    def synthesize(state: RagState) -> Dict[str, Any]:
        def doc_to_text(doc: Document, idx: int) -> str:
            source = doc.metadata.get("source", "unknown")
            kind = doc.metadata.get("kind", "file")
            sha = doc.metadata.get("sha", "")
            header = f"[{idx}] source={source} kind={kind}"
            if sha:
                header += f" sha={sha}"
            return f"{header}\n{doc.page_content[:1400]}"

        context = "\n\n".join(doc_to_text(d, i + 1) for i, d in enumerate(state["docs"]))
        answer = (answer_prompt | llm).invoke({"query": state["query"], "context": context}).content
        return {"answer": answer}

    graph = StateGraph(RagState)
    graph.add_node("classify", classify_query)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", synthesize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def ask(settings: Settings, query: str) -> str:
    app = build_graph(settings)
    result = app.invoke({"query": query, "query_type": "", "docs": [], "answer": ""})
    return result["answer"]
